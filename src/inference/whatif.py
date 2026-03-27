"""
IBC Counterfactual Simulator ("What-If" Engine) + Monte Carlo Simulation

Runs Monte Carlo simulation over input uncertainty to produce usable distributions.
Saves CSV (per-sample outputs) and JSON (summary stats).

Usage:
    python whatif.py --case-index 256 --mc 5000 --outdir outputs/mc
    python whatif.py --case-index 256 --mc 2000 --seed 7 --outdir outputs/mc

Notes:
    This is MONTE CARLO SIMULATION (randomize inputs), not MC dropout.
"""

import argparse
import json
import copy
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from ..models.physarum_gcn import PhysarumGCN, load_model
from ..utils.config import (
    inference_config, model_config, paths_config, data_config, NODE_TYPES
)
from ..data.graph_builder import (
    find_node, get_val, safe_float, safe_bool,
    COC_ALIGNMENT_THRESHOLD_PCT, TIMELINE_NORMAL_MAX_DAYS,
    TIMELINE_NORMALISATION_DAYS, DEFAULT_DAYS_MISSING,
)

# NODE_TYPES is the authoritative mapping defined in config.py.
# It is re-aliased here for clarity within this module.
NODE_TYPE_KEYS = NODE_TYPES


def build_graph_from_case(case_json: Dict) -> Data:
    """
    Build PyG Data object from case JSON (matches dataconverter.py logic).
    
    Args:
        case_json: Case data dictionary
    
    Returns:
        PyG Data object
    """
    # Locate the four logical sections of the case JSON.
    # Multiple key names are tried to handle schema variation across extraction runs.
    case_profile = find_node(case_json, ['node_1', 'case_profile', 'profile'])
    creditor_data = find_node(case_json, ['node_2', 'creditor', 'financial', 'dynamics'])
    promoter_data = find_node(case_json, ['node_3', 'promoter', 'behavior'])
    timeline_data = find_node(case_json, ['node_4', 'timeline', 'capital'])
    
    nodes = ['CASE_START']
    edges = []
    
    # Creditors
    creditor_buckets = {}
    total_claims = 0.0
    
    # Financial creditors — aggregate claims by creditor type
    fc = get_val(creditor_data.get('financial_creditors'))
    if isinstance(fc, list):
        for creditor in fc:
            if isinstance(creditor, dict):
                ctype = str(creditor.get('type', '')).lower()
                amount = safe_float(
                    creditor.get('amount_crores') or
                    creditor.get('amount')
                )
                if amount > 0:
                    if 'psu' in ctype or 'public' in ctype:
                        node_type = 'FC_PSU_BANK'
                    elif 'private' in ctype or 'bank' in ctype:
                        node_type = 'FC_PRIVATE_BANK'
                    elif 'nbfc' in ctype:
                        node_type = 'FC_NBFC'
                    elif 'arc' in ctype:
                        node_type = 'FC_ARC'
                    else:
                        node_type = 'FC_OTHER'

                    creditor_buckets[node_type] = creditor_buckets.get(node_type, 0) + amount
                    total_claims += amount

    # Operational creditors are pooled into a single OC_POOL node
    oc_amount = safe_float(get_val(creditor_data.get('operational_claims_total_crores')))
    if oc_amount > 0:
        creditor_buckets['OC_POOL'] = oc_amount
        total_claims += oc_amount

    # Fallback: use aggregate claim total if per-creditor breakdown is missing
    if total_claims == 0:
        total_claims = safe_float(get_val(creditor_data.get('total_admitted_claims')))
        if total_claims > 0:
            creditor_buckets['FC_OTHER'] = total_claims
    
    if not creditor_buckets:
        creditor_buckets['FC_OTHER'] = 0
        total_claims = 1.0
    
    for creditor_node, amount in creditor_buckets.items():
        nodes.append(creditor_node)
        # Edge weight = creditor's proportional claim share (minimum 0.01)
        weight = amount / total_claims if total_claims > 0 else 0.1
        edges.append(('CASE_START', creditor_node, max(0.01, weight)))

    # CoC node — top-creditor share > 40% means one party controls the vote
    top_share = safe_float(get_val(creditor_data.get('coc_voting_share_top_creditor_pct')))
    coc_node = 'COC_ALIGNED' if top_share > COC_ALIGNMENT_THRESHOLD_PCT else 'COC_FRAGMENTED'
    nodes.append(coc_node)
    for creditor_node in creditor_buckets:
        edges.append((creditor_node, coc_node, 1.0))

    # Promoter nodes — behaviour and Section 29A eligibility are separate axes
    is_cooperative = safe_bool(get_val(promoter_data.get('promoter_cooperating_with_rp')))
    submitted_plan = safe_bool(get_val(promoter_data.get('promoter_submitted_resolution_plan')))
    promoter_behavior_node = (
        'PROMOTER_COOPERATIVE'
        if (is_cooperative is True or submitted_plan is True)
        else 'PROMOTER_HOSTILE'
    )

    is_ineligible = safe_bool(get_val(promoter_data.get('promoter_is_section_29a_ineligible')))
    promoter_eligibility_node = (
        'PROMOTER_29A_BLOCKED' if is_ineligible is True else 'PROMOTER_29A_ELIGIBLE'
    )

    nodes.extend([promoter_behavior_node, promoter_eligibility_node])
    edges.append((coc_node, promoter_behavior_node, 1.0))
    edges.append((promoter_behavior_node, promoter_eligibility_node, 1.0))

    # Timeline node — 330 days separates within-window from substantially delayed cases
    days = safe_float(get_val(timeline_data.get('total_days_in_process')))
    if days == 0:
        days = DEFAULT_DAYS_MISSING  # dataset median when field is missing

    timeline_node = 'TIMELINE_NORMAL' if days < TIMELINE_NORMAL_MAX_DAYS else 'TIMELINE_EXTENDED'
    nodes.append(timeline_node)
    edges.append((promoter_eligibility_node, timeline_node, 1.0))

    # Outcome nodes — equal weights (0.33 each); the prediction comes from graph structure
    nodes.extend(['RESOLUTION_STRATEGIC', 'RESOLUTION_PROMOTER', 'LIQUIDATION'])
    edges.append((timeline_node, 'RESOLUTION_STRATEGIC', 0.33))
    edges.append((timeline_node, 'RESOLUTION_PROMOTER', 0.33))
    edges.append((timeline_node, 'LIQUIDATION', 0.33))

    # Node features (22-dim): 18 one-hot type dims + 4 continuous case-level dims
    process_type = str(get_val(case_profile.get('process_type')) or "").lower()
    company_size = str(get_val(case_profile.get('company_size')) or "").lower()

    feat_is_ppirp = 1.0 if "ppirp" in process_type else 0.0       # dim 18
    feat_is_msme = 1.0 if "msme" in company_size else 0.0          # dim 19
    feat_log_claims = np.log1p(total_claims) / 10.0                 # dim 20
    feat_timeline_urgency = max(0.1, 1.0 - (days / TIMELINE_NORMALISATION_DAYS))  # dim 21

    node_features = []
    for node in nodes:
        feat = np.zeros(22, dtype=np.float32)
        # Dimensions 0–17: one-hot node type
        if node in NODE_TYPE_KEYS:
            feat[NODE_TYPE_KEYS[node]] = 1.0
        # Dimensions 18–21: case-level context (same value for every node in the graph)
        feat[18] = feat_is_ppirp
        feat[19] = feat_is_msme
        feat[20] = feat_log_claims
        feat[21] = feat_timeline_urgency
        node_features.append(feat)
    
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    edge_indices, edge_attrs = [], []
    for u, v, w in edges:
        edge_indices.append([node_to_idx[u], node_to_idx[v]])
        edge_attrs.append([w])
    
    return Data(
        x=x,
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attrs, dtype=torch.float),
    )


def predict_probs(model: PhysarumGCN, graph: Data, device: str) -> np.ndarray:
    """
    Predict class probabilities for a graph.
    
    Args:
        model: Trained model
        graph: PyG Data object
        device: Device to run on
    
    Returns:
        Class probabilities array [num_classes]
    """
    model.eval()
    with torch.no_grad():
        graph = graph.to(device)
        logits, _ = model(graph)
        # FIX: Use .cpu() before .numpy() for CUDA tensors
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def jitter_case_inputs(case_json: Dict, rng: random.Random) -> Dict:
    """
    Monte Carlo input uncertainty model.
    
    Conservative randomization:
    - Creditor amounts: lognormal multiplier (sigma=0.20)
    - Top creditor CoC share: normal noise (sd=4), clipped [0, 100]
    - Days in process: normal noise (sd=25), clipped [30, 1200]
    - Promoter cooperation: small flip prob (p=0.03) if present
    - 29A ineligibility: small flip prob (p=0.02) if present
    
    Args:
        case_json: Original case data
        rng: Random number generator
    
    Returns:
        Jittered case data
    """
    case = copy.deepcopy(case_json)

    creditor_data = find_node(case, ['node_2', 'creditor'])
    promoter_data = find_node(case, ['node_3', 'promoter'])
    timeline_data = find_node(case, ['node_4', 'timeline'])

    # Creditor amount jitter — lognormal so amounts stay positive
    fc = get_val(creditor_data.get('financial_creditors'))
    if isinstance(fc, list):
        for cred in fc:
            if isinstance(cred, dict):
                amt = safe_float(cred.get('amount_crores') or cred.get('amount'))
                if amt > 0:
                    mult = float(np.exp(rng.gauss(0.0, 0.20)))
                    cred['amount_crores'] = max(0.0, amt * mult)

    # CoC share jitter — additive Gaussian, clipped to valid percentage range
    t = safe_float(get_val(creditor_data.get('coc_voting_share_top_creditor_pct')))
    if t > 0:
        t_jittered = float(np.clip(t + rng.gauss(0.0, 4.0), 0.0, 100.0))
        creditor_data['coc_voting_share_top_creditor_pct'] = {'value': t_jittered, 'confidence': 1.0}

    # Days jitter — additive Gaussian, clipped to plausible IBC range [30, 1200]
    d = safe_float(get_val(timeline_data.get('total_days_in_process')))
    if d == 0:
        d = DEFAULT_DAYS_MISSING
    d_jittered = float(np.clip(d + rng.gauss(0.0, 25.0), 30.0, 1200.0))
    timeline_data['total_days_in_process'] = {'value': d_jittered, 'confidence': 1.0}

    # Promoter cooperation flip — small probability to simulate extraction uncertainty
    coop = get_val(promoter_data.get('promoter_cooperating_with_rp'))
    if coop is not None and rng.random() < 0.03:
        new_coop = not coop if isinstance(coop, bool) else False
        promoter_data['promoter_cooperating_with_rp'] = {'value': new_coop, 'confidence': 0.8}

    # 29A ineligibility flip — lower flip rate as this is more stable/verifiable
    ineligible = get_val(promoter_data.get('promoter_is_section_29a_ineligible'))
    if ineligible is not None and rng.random() < 0.02:
        new_ineligible = not ineligible if isinstance(ineligible, bool) else False
        promoter_data['promoter_is_section_29a_ineligible'] = {'value': new_ineligible, 'confidence': 0.8}
    
    return case


def run_monte_carlo(
    model: PhysarumGCN,
    case_json: Dict,
    n_samples: int,
    rng: random.Random,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Monte Carlo simulation.
    
    Args:
        model: Trained model
        case_json: Base case data
        n_samples: Number of MC samples
        rng: Random number generator
        device: Device to run on
    
    Returns:
        Tuple of (all_probs, base_probs)
    """
    # Baseline (no jitter)
    base_graph = build_graph_from_case(case_json)
    base_probs = predict_probs(model, base_graph, device)
    
    all_probs = []
    for _ in range(n_samples):
        jittered_case = jitter_case_inputs(case_json, rng)
        graph = build_graph_from_case(jittered_case)
        probs = predict_probs(model, graph, device)
        all_probs.append(probs)
    
    return np.array(all_probs), base_probs


def main():
    parser = argparse.ArgumentParser(description="IBC What-If Monte Carlo Simulator")
    parser.add_argument("--case-index", type=int, default=inference_config.default_case_index,
                        help="Index of case to analyze")
    parser.add_argument("--mc", type=int, default=inference_config.mc_samples,
                        help="Number of Monte Carlo samples")
    parser.add_argument("--seed", type=int, default=inference_config.mc_seed,
                        help="Random seed")
    parser.add_argument("--outdir", type=str, default=str(paths_config.mc_output_dir),
                        help="Output directory")
    parser.add_argument("--model", type=str, default=str(paths_config.final_model_path),
                        help="Model path")
    parser.add_argument("--data", type=str, default=str(data_config.extracted_json_path),
                        help="Data JSON path")
    
    args = parser.parse_args()
    
    # Setup
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.model}...")
    model = load_model(
        model_path=args.model,
        input_dim=model_config.input_dim_ibc,
        hidden_dim=model_config.hidden_dim,
        num_classes=model_config.num_classes,
        device=device
    )
    
    # Load data
    print(f"Loading data from {args.data}...")
    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get case - handle both direct data and wrapped data
    case_data = data[args.case_index]
    case = case_data.get('data', case_data)  # FIX: Flexible data access
    
    case_number = case.get('case_metadata', {}).get('case_number', {}).get('value', f'case_{args.case_index}')
    print(f"\nAnalyzing case: {case_number}")
    
    # Setup RNG
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Run Monte Carlo
    print(f"Running {args.mc} Monte Carlo samples...")
    all_probs, base_probs = run_monte_carlo(model, case, args.mc, rng, device)
    
    # Statistics
    classes = inference_config.class_names
    
    print("\n" + "="*60)
    print("BASELINE PREDICTION (no jitter)")
    print("="*60)
    for i, cls in enumerate(classes):
        print(f"  {cls}: {base_probs[i]:.1%}")
    pred_class = classes[base_probs.argmax()]
    print(f"  -> Predicted: {pred_class}")
    
    print("\n" + "="*60)
    print("MONTE CARLO RESULTS (input uncertainty)")
    print("="*60)
    
    summary = {}
    for i, cls in enumerate(classes):
        probs = all_probs[:, i]
        summary[cls] = {
            "mean": float(np.mean(probs)),
            "std": float(np.std(probs)),
            "median": float(np.median(probs)),
            "q25": float(np.percentile(probs, 25)),
            "q75": float(np.percentile(probs, 75)),
            "min": float(np.min(probs)),
            "max": float(np.max(probs)),
        }
        print(f"\n{cls}:")
        print(f"  Mean:   {summary[cls]['mean']:.1%}")
        print(f"  Std:    {summary[cls]['std']:.1%}")
        print(f"  95% CI: [{summary[cls]['q25']:.1%}, {summary[cls]['q75']:.1%}]")
    
    # Most likely outcome across all samples
    mc_predictions = all_probs.argmax(axis=1)
    mode_class_idx = int(np.bincount(mc_predictions).argmax())
    mode_class = classes[mode_class_idx]
    mode_confidence = float(np.mean(mc_predictions == mode_class_idx))
    
    print(f"\n{'='*60}")
    print(f"MOST LIKELY OUTCOME (across {args.mc} samples)")
    print("="*60)
    print(f"  {mode_class} ({mode_confidence:.1%} of samples)")
    
    # Save results
    prefix = f"case{args.case_index}_N{args.mc}"
    
    # CSV: per-sample probabilities
    csv_path = outdir / f"mc_results_{prefix}.csv"
    with open(csv_path, 'w') as f:
        f.write("sample," + ",".join(classes) + "\n")
        for i, probs in enumerate(all_probs):
            f.write(f"{i}," + ",".join(f"{p:.6f}" for p in probs) + "\n")
    print(f"\n✓ Saved per-sample results to: {csv_path}")
    
    # JSON: summary stats
    json_path = outdir / f"mc_summary_{prefix}.json"
    output = {
        "case_index": args.case_index,
        "case_number": case_number,
        "n_samples": args.mc,
        "seed": args.seed,
        "baseline": {
            cls: float(base_probs[i]) for i, cls in enumerate(classes)
        },
        "monte_carlo": summary,
        "most_likely_outcome": {
            "class": mode_class,
            "confidence": mode_confidence
        }
    }
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"✓ Saved summary statistics to: {json_path}")


if __name__ == "__main__":
    main()
