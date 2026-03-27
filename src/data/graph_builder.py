"""
Graph Builder - Converts extracted case JSON into PyTorch Geometric Data objects.

Each IBC case is represented as a heterogeneous graph where:
- Nodes represent entities (case start, creditors by type, CoC, promoter, timeline, outcomes)
- Edges represent relationships (creditor → CoC → promoter → timeline → outcomes)
- Node features are 22-dimensional: 18 one-hot type flags + 4 continuous case-level features

Node feature layout (22 dimensions):
    [0-17]  One-hot encoding of node type (see NODE_TYPE_KEYS in config.py)
    [18]    Process type flag: 1.0 if PPIRP (pre-packaged), 0.0 if standard CIRP
    [19]    Company size flag: 1.0 if MSME, 0.0 otherwise
    [20]    Log-scaled total admitted claims (log1p(crores) / 10.0, normalised)
    [21]    Timeline urgency: 1.0 - (days / 660), clipped to [0.1, 1.0]

Outcome labels (for training):
    0 - Strategic Resolution  (third-party acquirer approved resolution plan)
    1 - Promoter Re-entry     (promoter re-acquired via Section 12A withdrawal)
    2 - Liquidation           (NCLT ordered liquidation)

Thresholds used in graph construction:
    - CoC concentration: top-creditor voting share > 40% → COC_ALIGNED, else COC_FRAGMENTED
    - Timeline classification: total days < 330 → TIMELINE_NORMAL, else TIMELINE_EXTENDED
    - Timeline normalisation denominator: 660 days (≈ statutory 270-day limit + typical overrun)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from ..utils.config import NODE_TYPES


# ─── Outcome label mapping ────────────────────────────────────────────────────
# Maps substrings found in the raw JSON outcome field to integer class labels.
OUTCOME_LABEL_MAP: Dict[str, int] = {
    # Class 0 – Strategic Resolution
    "strategic": 0,
    "resolution_plan": 0,
    "resolution plan": 0,
    "approved": 0,
    # Class 1 – Promoter Re-entry (Section 12A withdrawal)
    "promoter": 1,
    "section_12a": 1,
    "section 12a": 1,
    "withdrawal": 1,
    # Class 2 – Liquidation
    "liquidation": 2,
}

# Classification thresholds (must match whatif.py and training data converter)
COC_ALIGNMENT_THRESHOLD_PCT = 40.0   # Top creditor share above this → aligned CoC
TIMELINE_NORMAL_MAX_DAYS = 330       # Days below this → normal timeline
TIMELINE_NORMALISATION_DAYS = 660.0  # Denominator for timeline urgency feature
DEFAULT_DAYS_MISSING = 400.0         # Fallback when days field is absent


# ─── Low-level JSON helpers ───────────────────────────────────────────────────

def find_node(data: Dict, keys: List[str]) -> Dict:
    """
    Return the first sub-dict found by trying each key in order.

    Extracted case JSON can use different key names depending on the LLM
    extraction prompt version (e.g. 'node_1', 'case_profile', 'profile').
    This helper abstracts that variation away.

    Args:
        data: Top-level case dictionary.
        keys: Candidate keys to try, in priority order.

    Returns:
        The matched sub-dict, or {} if none of the keys exist.
    """
    for key in keys:
        if key in data and isinstance(data[key], dict):
            return data[key]
    return {}


def get_val(item: Any) -> Any:
    """
    Unwrap a confidence-annotated value dict produced by the LLM extractor.

    The LLM extraction pipeline wraps every field as::

        {"value": <actual_value>, "confidence": 0.9}

    This function transparently handles both wrapped and plain values so the
    rest of the code doesn't need to check which format it received.

    Args:
        item: Either a plain value or a dict with a 'value' key.

    Returns:
        The unwrapped value, or the original item if it is not a wrapped dict.
    """
    if isinstance(item, dict) and "value" in item:
        return item["value"]
    return item


def safe_float(x: Any) -> float:
    """
    Convert *x* to float, returning 0.0 on any failure.

    Handles None, empty string, and non-numeric types without raising.

    Args:
        x: Value to convert.

    Returns:
        Float representation of *x*, or 0.0 if conversion fails.
    """
    try:
        val = get_val(x)
        if val is None:
            return 0.0
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def safe_bool(x: Any) -> Optional[bool]:
    """
    Convert *x* to bool, returning None if the value is absent or ambiguous.

    Recognises common string representations ("true"/"false"/"yes"/"no") in
    addition to native Python booleans and integers.

    Args:
        x: Value to interpret.

    Returns:
        True, False, or None if the value cannot be meaningfully interpreted.
    """
    val = get_val(x)
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        lower = val.strip().lower()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
    return None


# ─── Graph construction ───────────────────────────────────────────────────────

def build_graph_from_case(case_json: Dict) -> Data:
    """
    Build a PyG Data object from a single extracted case dictionary.

    Graph topology
    --------------
    CASE_START
      └─► FC_* / OC_POOL  (weight = creditor's share of total claims)
            └─► COC_ALIGNED | COC_FRAGMENTED  (weight = 1.0)
                  └─► PROMOTER_COOPERATIVE | PROMOTER_HOSTILE  (weight = 1.0)
                        └─► PROMOTER_29A_ELIGIBLE | PROMOTER_29A_BLOCKED  (weight = 1.0)
                              └─► TIMELINE_NORMAL | TIMELINE_EXTENDED  (weight = 1.0)
                                    ├─► RESOLUTION_STRATEGIC  (weight = 0.33)
                                    ├─► RESOLUTION_PROMOTER   (weight = 0.33)
                                    └─► LIQUIDATION           (weight = 0.33)

    The three outcome nodes are always present but their class label is not
    encoded here — it is stored in the ``outcome`` attribute of the Data object
    when created by :func:`convert_json_to_graphs`.

    Args:
        case_json: Extracted case dictionary (top-level keys depend on the
                   extraction schema version).

    Returns:
        PyG Data object with ``x`` (node features) and ``edge_index`` /
        ``edge_attr`` (graph structure).
    """
    # Locate the four logical sections of the case JSON.
    # Multiple key names are tried to handle schema evolution across extraction runs.
    case_profile = find_node(case_json, ["node_1", "case_profile", "profile"])
    creditor_data = find_node(case_json, ["node_2", "creditor", "financial", "dynamics"])
    promoter_data = find_node(case_json, ["node_3", "promoter", "behavior"])
    timeline_data = find_node(case_json, ["node_4", "timeline", "capital"])

    nodes: List[str] = ["CASE_START"]
    edges: List[tuple] = []  # (source_node, target_node, edge_weight)

    # ── Creditor nodes ─────────────────────────────────────────────────────
    # Aggregate claims by creditor type to form one node per type bucket.
    creditor_buckets: Dict[str, float] = {}
    total_claims = 0.0

    financial_creditors = get_val(creditor_data.get("financial_creditors"))
    if isinstance(financial_creditors, list):
        for creditor in financial_creditors:
            if not isinstance(creditor, dict):
                continue
            ctype = str(creditor.get("type", "")).lower()
            amount = safe_float(
                creditor.get("amount_crores") or creditor.get("amount")
            )
            if amount <= 0:
                continue

            # Map creditor type string to a node type key
            if "psu" in ctype or "public" in ctype:
                node_type = "FC_PSU_BANK"
            elif "private" in ctype or "bank" in ctype:
                node_type = "FC_PRIVATE_BANK"
            elif "nbfc" in ctype:
                node_type = "FC_NBFC"
            elif "arc" in ctype:
                node_type = "FC_ARC"
            else:
                node_type = "FC_OTHER"

            creditor_buckets[node_type] = creditor_buckets.get(node_type, 0.0) + amount
            total_claims += amount

    # Operational creditors are pooled into a single OC_POOL node
    oc_amount = safe_float(get_val(creditor_data.get("operational_claims_total_crores")))
    if oc_amount > 0:
        creditor_buckets["OC_POOL"] = oc_amount
        total_claims += oc_amount

    # Fallback: use aggregate claim total if per-creditor breakdown is missing
    if total_claims == 0:
        total_claims = safe_float(get_val(creditor_data.get("total_admitted_claims")))
        if total_claims > 0:
            creditor_buckets["FC_OTHER"] = total_claims

    # Ensure at least one creditor node exists
    if not creditor_buckets:
        creditor_buckets["FC_OTHER"] = 0.0
        total_claims = 1.0

    for creditor_node, amount in creditor_buckets.items():
        nodes.append(creditor_node)
        # Edge weight = creditor's proportional claim share (minimum 0.01 to avoid zero weights)
        weight = amount / total_claims if total_claims > 0 else 0.1
        edges.append(("CASE_START", creditor_node, max(0.01, weight)))

    # ── CoC node ───────────────────────────────────────────────────────────
    # A top-creditor voting share above 40% means one party can block resolutions,
    # capturing the CoC power dynamics that research shows drives outcomes.
    top_share = safe_float(get_val(creditor_data.get("coc_voting_share_top_creditor_pct")))
    coc_node = "COC_ALIGNED" if top_share > COC_ALIGNMENT_THRESHOLD_PCT else "COC_FRAGMENTED"
    nodes.append(coc_node)
    for creditor_node in creditor_buckets:
        edges.append((creditor_node, coc_node, 1.0))

    # ── Promoter nodes ─────────────────────────────────────────────────────
    is_cooperative = safe_bool(get_val(promoter_data.get("promoter_cooperating_with_rp")))
    submitted_plan = safe_bool(get_val(promoter_data.get("promoter_submitted_resolution_plan")))
    # Cooperative if either cooperating with RP or submitted a resolution plan
    promoter_behavior_node = (
        "PROMOTER_COOPERATIVE"
        if (is_cooperative is True or submitted_plan is True)
        else "PROMOTER_HOSTILE"
    )

    is_ineligible = safe_bool(get_val(promoter_data.get("promoter_is_section_29a_ineligible")))
    promoter_eligibility_node = (
        "PROMOTER_29A_BLOCKED" if is_ineligible is True else "PROMOTER_29A_ELIGIBLE"
    )

    nodes.extend([promoter_behavior_node, promoter_eligibility_node])
    edges.append((coc_node, promoter_behavior_node, 1.0))
    edges.append((promoter_behavior_node, promoter_eligibility_node, 1.0))

    # ── Timeline node ──────────────────────────────────────────────────────
    # 330 days separates cases resolved within the original statutory window
    # (~270 days + one standard extension) from substantially delayed ones.
    days = safe_float(get_val(timeline_data.get("total_days_in_process")))
    if days == 0:
        days = DEFAULT_DAYS_MISSING  # ~400 days is close to the dataset median

    timeline_node = "TIMELINE_NORMAL" if days < TIMELINE_NORMAL_MAX_DAYS else "TIMELINE_EXTENDED"
    nodes.append(timeline_node)
    edges.append((promoter_eligibility_node, timeline_node, 1.0))

    # ── Outcome nodes ──────────────────────────────────────────────────────
    # All three outcome nodes are always added. Equal weights (0.33 each) signal
    # that the graph structure itself, not these edges, determines the prediction.
    nodes.extend(["RESOLUTION_STRATEGIC", "RESOLUTION_PROMOTER", "LIQUIDATION"])
    edges.append((timeline_node, "RESOLUTION_STRATEGIC", 0.33))
    edges.append((timeline_node, "RESOLUTION_PROMOTER", 0.33))
    edges.append((timeline_node, "LIQUIDATION", 0.33))

    # ── Node features (22-dim) ─────────────────────────────────────────────
    process_type = str(get_val(case_profile.get("process_type")) or "").lower()
    company_size = str(get_val(case_profile.get("company_size")) or "").lower()

    # Feature [18]: PPIRP flag (pre-packaged insolvency, faster track)
    feat_is_ppirp = 1.0 if "ppirp" in process_type else 0.0
    # Feature [19]: MSME flag (smaller companies have different resolution dynamics)
    feat_is_msme = 1.0 if "msme" in company_size else 0.0
    # Feature [20]: Log-scaled total claims (log1p prevents zero issues; /10 normalises to ~[0,1])
    feat_log_claims = np.log1p(total_claims) / 10.0
    # Feature [21]: Timeline urgency — higher means case resolved faster than typical
    feat_timeline_urgency = max(0.1, 1.0 - (days / TIMELINE_NORMALISATION_DAYS))

    node_features = []
    for node in nodes:
        feat = np.zeros(22, dtype=np.float32)
        # Dimensions 0–17: one-hot node type encoding
        if node in NODE_TYPES:
            feat[NODE_TYPES[node]] = 1.0
        # Dimensions 18–21: case-level continuous features (same for all nodes in a graph)
        feat[18] = feat_is_ppirp
        feat[19] = feat_is_msme
        feat[20] = feat_log_claims
        feat[21] = feat_timeline_urgency
        node_features.append(feat)

    x = torch.tensor(np.array(node_features), dtype=torch.float)
    node_to_idx = {name: i for i, name in enumerate(nodes)}

    edge_indices, edge_attrs = [], []
    for src, dst, weight in edges:
        edge_indices.append([node_to_idx[src], node_to_idx[dst]])
        edge_attrs.append([weight])

    return Data(
        x=x,
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attrs, dtype=torch.float),
    )


def _extract_outcome_label(case_json: Dict) -> Optional[int]:
    """
    Extract the integer outcome class label (0/1/2) from a case dict.

    Tries several common field names and substring-matches the value against
    :data:`OUTCOME_LABEL_MAP`. Returns None if no recognisable outcome is found
    (those cases are skipped during conversion).

    Args:
        case_json: Top-level case dictionary.

    Returns:
        0 (Strategic), 1 (Promoter), 2 (Liquidation), or None.
    """
    candidate_keys = ["outcome", "final_outcome", "result", "decision", "case_outcome"]
    for key in candidate_keys:
        raw = case_json.get(key)
        if raw is None:
            # Also check inside case_metadata
            meta = case_json.get("case_metadata", {})
            raw = meta.get(key)
        if raw is None:
            continue

        val_str = str(get_val(raw)).lower().strip()
        for substring, label in OUTCOME_LABEL_MAP.items():
            if substring in val_str:
                return label

    return None


# ─── Batch conversion ─────────────────────────────────────────────────────────

def convert_json_to_graphs(
    input_path: str,
    output_path: str,
    failed_path: Optional[str] = None,
) -> None:
    """
    Convert a JSON file of extracted IBC cases into a list of PyG Data objects
    and save it as a ``.pt`` file for use in training.

    The input file should be a JSON array where each element is a case dict
    as produced by the LLM extraction step. Cases whose outcome label cannot
    be determined are skipped and optionally written to *failed_path*.

    Args:
        input_path:  Path to the extracted cases JSON file.
        output_path: Path for the output ``.pt`` file (list of Data objects).
        failed_path: Optional path to write a JSON file listing cases that
                     could not be converted (for debugging).

    Raises:
        FileNotFoundError: If *input_path* does not exist.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading cases from {input_path}...")
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        # Support both a bare list and a dict with a 'cases' key
        raw_data = raw_data.get("cases", [raw_data])

    graphs: List[Data] = []
    failed: List[Dict] = []

    for idx, item in enumerate(raw_data):
        # Unwrap optional outer wrapper (e.g. {"data": {...}, "meta": {...}})
        case = item.get("data", item) if isinstance(item, dict) else item

        try:
            outcome = _extract_outcome_label(case)
            if outcome is None:
                failed.append({"index": idx, "reason": "outcome_not_found"})
                continue

            graph = build_graph_from_case(case)
            graph.outcome = torch.tensor([outcome], dtype=torch.long)
            graphs.append(graph)

        except Exception as exc:
            failed.append({"index": idx, "reason": str(exc)})

    print(f"Converted {len(graphs)} cases  |  skipped {len(failed)}")

    # Save graph list
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, output_path)
    print(f"Saved graph data to {output_path}")

    # Optionally save failed cases for inspection
    if failed_path and failed:
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        print(f"Saved {len(failed)} failed cases to {failed_path}")
