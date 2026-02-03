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
    inference_config, paths_config, data_config, NODE_TYPES
)
from ..data.graph_builder import find_node, get_val, safe_float, safe_bool


# Node type mapping (must match training)
NODE_TYPE_KEYS = {
    'CASE_START': 0, 'FC_PSU_BANK': 1, 'FC_PRIVATE_BANK': 2, 'FC_NBFC': 3,
    'FC_ARC': 4, 'FC_OTHER': 5, 'OC_POOL': 6, 'COC_ALIGNED': 7,
    'COC_FRAGMENTED': 8, 'PROMOTER_COOPERATIVE': 9, 'PROMOTER_HOSTILE': 10,
    'PROMOTER_29A_ELIGIBLE': 11, 'PROMOTER_29A_BLOCKED': 12,
    'TIMELINE_NORMAL': 13, 'TIMELINE_EXTENDED': 14,
    'RESOLUTION_STRATEGIC': 15, 'RESOLUTION_PROMOTER': 16, 'LIQUIDATION': 17,
}


def build_graph_from_case(case_json: Dict) -> Data:
    """
    Build PyG Data object from case JSON (matches dataconverter.py logic).
    
    Args:
        case_json: Case data dictionary
    
    Returns:
        PyG Data object
    """
    # Extract nodes
    n1 = find_node(case_json, ['node_1', 'case_profile', 'profile'])
    n2 = find_node(case_json, ['node_2', 'creditor', 'financial', 'dynamics'])
    n3 = find_node(case_json, ['node_3', 'promoter', 'behavior'])
    n4 = find_node(case_json, ['node_4', 'timeline', 'capital'])
    
    nodes = ['CASE_START']
    edges = []
    
    # Creditors
    creditor_buckets = {}
    total_claims = 0.0
    
    # Financial creditors
    fc = get_val(n2.get('financial_creditors'))
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
    
    # Operational creditors
    oc_amount = safe_float(get_val(n2.get('operational_claims_total_crores')))
    if oc_amount > 0:
        creditor_buckets['OC_POOL'] = oc_amount
        total_claims += oc_amount
    
    # Fallback
    if total_claims == 0:
        total_claims = safe_float(get_val(n2.get('total_admitted_claims')))
        if total_claims > 0:
            creditor_buckets['FC_OTHER'] = total_claims
    
    if not creditor_buckets:
        creditor_buckets['FC_OTHER'] = 0
        total_claims = 1.0
    
    for c_node, amount in creditor_buckets.items():
        nodes.append(c_node)
        weight = amount / total_claims if total_claims > 0 else 0.1
        edges.append(('CASE_START', c_node, max(0.01, weight)))
    
    # CoC
    top_share = safe_float(get_val(n2.get('coc_voting_share_top_creditor_pct')))
    coc_node = 'COC_ALIGNED' if top_share > 40 else 'COC_FRAGMENTED'
    nodes.append(coc_node)
    
    for c_node in creditor_buckets:
        edges.append((c_node, coc_node, 1.0))
    
    # Promoter
    is_coop = safe_bool(get_val(n3.get('promoter_cooperating_with_rp')))
    submitted = safe_bool(get_val(n3.get('promoter_submitted_resolution_plan')))
    promo_node = 'PROMOTER_COOPERATIVE' if (is_coop is True or submitted is True) else 'PROMOTER_HOSTILE'
    
    is_ineligible = safe_bool(get_val(n3.get('promoter_is_section_29a_ineligible')))
    elig_node = 'PROMOTER_29A_BLOCKED' if is_ineligible is True else 'PROMOTER_29A_ELIGIBLE'
    
    nodes.extend([promo_node, elig_node])
    edges.append((coc_node, promo_node, 1.0))
    edges.append((promo_node, elig_node, 1.0))
    
    # Timeline
    days = safe_float(get_val(n4.get('total_days_in_process')))
    if days == 0:
        days = 400.0
    
    time_node = 'TIMELINE_NORMAL' if days < 330 else 'TIMELINE_EXTENDED'
    nodes.append(time_node)
    edges.append((elig_node, time_node, 1.0))
    
    # Outcomes
    nodes.extend(['RESOLUTION_STRATEGIC', 'RESOLUTION_PROMOTER', 'LIQUIDATION'])
    edges.append((time_node, 'RESOLUTION_STRATEGIC', 0.33))
    edges.append((time_node, 'RESOLUTION_PROMOTER', 0.33))
    edges.append((time_node, 'LIQUIDATION', 0.33))
    
    # Features
    node_features = []
    p_type = str(get_val(n1.get('process_type')) or "").lower()
    c_size = str(get_val(n1.get('company_size')) or "").lower()
    
    f_process = 1.0 if "ppirp" in p_type else 0.0
    f_msme = 1.0 if "msme" in c_size else 0.0
    f_claims = np.log1p(total_claims) / 10.0
    f_timeline = max(0.1, 1.0 - (days / 660.0))
    
    for node in nodes:
        feat = np.zeros(22, dtype=np.float32)
        if node in NODE_TYPE_KEYS:
            feat[NODE_TYPE_KEYS[node]] = 1.0
        feat[18], feat[19], feat[20], feat[21] = f_process, f_msme, f_claims, f_timeline
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
    
    n2 = find_node(case, ['node_2', 'creditor'])
    n3 = find_node(case, ['node_3', 'promoter'])
    n4 = find_node(case, ['node_4', 'timeline'])
    
    # Creditor amount jitter
    fc = get_val(n2.get('financial_creditors'))
    if isinstance(fc, list):
        for cred in fc:
            if isinstance(cred, dict):
                amt = cred.get('amount_crores') or cred.get('amount')
                amt = safe_float(amt)
                if amt > 0:
                    # FIX: Use rng instead of np.random
                    mult = float(np.exp(rng.gauss(0.0, 0.20)))
                    new_amt = max(0.0, amt * mult)
                    if 'amount_crores' in cred:
                        cred['amount_crores'] = new_amt
                    else:
                        cred['amount_crores'] = new_amt
    
    # CoC share jitter
    t = get_val(n2.get('coc_voting_share_top_creditor_pct'))
    t = safe_float(t)
    if t > 0:
        # FIX: Use rng instead of np.random
        t2 = float(np.clip(t + rng.gauss(0.0, 4.0), 0.0, 100.0))
        n2['coc_voting_share_top_creditor_pct'] = {'value': t2, 'confidence': 1.0}
    
    # Days jitter
    d = safe_float(get_val(n4.get('total_days_in_process')))
    if d == 0:
        d = 400.0
    # FIX: Use rng instead of np.random
    d2 = float(np.clip(d + rng.gauss(0.0, 25.0), 30.0, 1200.0))
    n4['total_days_in_process'] = {'value': d2, 'confidence': 1.0}
    
    # Promoter cooperation flip
    coop = get_val(n3.get('promoter_cooperating_with_rp'))
    if coop is not None and rng.random() < 0.03:
        new_coop = not coop if isinstance(coop, bool) else False
        n3['promoter_cooperating_with_rp'] = {'value': new_coop, 'confidence': 0.8}
    
    # 29A ineligibility flip
    ineligible = get_val(n3.get('promoter_is_section_29a_ineligible'))
    if ineligible is not None and rng.random() < 0.02:
        new_ineligible = not ineligible if isinstance(ineligible, bool) else False
        n3['promoter_is_section_29a_ineligible'] = {'value': new_ineligible, 'confidence': 0.8}
    
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
    
    # Load model - FIX: Device placement before loading weights
    print(f"Loading model from {args.model}...")
    model = load_model(
        model_path=args.model,
        input_dim=22,
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
