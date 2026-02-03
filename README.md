# IBC Case Outcome Prediction

A computational census of 4,820 IBC cases analyzing whether India's insolvency framework is predictable or arbitrary.

## The Question

India's net FDI collapsed from $10 billion to $353 million in FY2024-25. Foreign investors cite regulatory uncertainty and insolvency unpredictability among their concerns. The conventional diagnosis: IBC outcomes are arbitrary—bench assignment matters more than case merits.

**This research tests that diagnosis.**

## Key Finding

IBC outcomes are **71.9% predictable** from observable case characteristics. If outcomes were truly arbitrary, predictability would approach the 35.9% majority-class baseline.

The problem isn't judicial inconsistency—it's **information asymmetry**. Sophisticated players (PSU banks, ARCs, distressed debt funds) can read patterns that remain opaque to smaller creditors and foreign investors.

## Results

| Metric | Value |
|--------|-------|
| Dataset | 4,820 cases (100% of resolved IBC proceedings through Sept 2024) |
| Weighted F1 | 71.9% |
| Accuracy | 70.7% |
| Majority Baseline | 35.9% |

### Outcome Distribution

| Class | Count | Proportion |
|-------|-------|------------|
| Strategic Resolution | 2,119 | 44.0% |
| Promoter Re-entry | 179 | 3.7% |
| Liquidation | 2,522 | 52.3% |

### Per-Class Performance

| Outcome | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| Strategic Resolution | 69.0% | 70.5% | 0.70 |
| Promoter Re-entry | 30.9% | 94.4% | 0.47 |
| Liquidation | 83.0% | 69.3% | 0.75 |

## Counterfactual Analysis

Monte Carlo simulations (N=5,000) reveal what sophisticated players know:

| Scenario | Δ Strategic | Δ Promoter | Δ Liquidation |
|----------|-------------|------------|---------------|
| Reduce delay (→180d) | −61.5% | ≈0% | **+61.5%** |
| Promoter cooperates | −57.1% | **+69.5%** | −12.4% |
| Debt → ₹50,000 Cr | +1.5% | +0.5% | −1.9% |

**Key insights:**
- Fast resolution → higher liquidation (quick cases often lack viable resolution)
- Promoter cooperation is the single strongest lever (+69.5%)
- Debt size barely matters (<2% shift)

## Model Architecture

3-layer Graph Convolutional Network (Kipf & Welling, 2017):

- **Architecture**: 22 → 128 → 128 → 64 dimensions
- **Regularization**: Dropout (0.4), class-weighted loss [1.0, 8.0, 1.5]
- **Optimizer**: AdamW (lr = 5×10⁻⁴)

### Graph Representation

Each IBC case is modeled as a graph where:
- **Nodes**: Financial creditors, operational creditors, CoC dynamics, promoter behavior
- **Edges**: Relationships weighted by financial exposure
- **Features**: 22 dimensions (financial ratios, creditor composition, case attributes)

## Data Pipeline

```
IBBI/NCLT Orders → OCR (PaddleOCR) → LLM Extraction → Graph Construction → GCN
```

| Component | Description |
|-----------|-------------|
| `datascraper/` | Downloads NCLT judgments from IBBI |
| `pipelinefrocr/` | OCR extraction using PaddleOCR |
| `src/models/` | GCN architecture |
| `src/training/` | Training loop with early stopping |
| `src/inference/` | Monte Carlo counterfactual analysis |

## Installation

```bash
git clone https://github.com/THE-PRV/ibc-case-prediction-gnn.git
cd ibc-case-prediction-gnn

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### Training
```bash
python src/training/train.py
```

### Counterfactual Analysis
```bash
python src/inference/whatif.py --case-index 256 --mc 5000 --outdir outputs_mc
```

## Policy Implication

The paper proposes expanding **Information Utilities** (IBC Sections 214–215) to democratize predictive insights:
- Publish outcome probability distributions for pending cases
- Allow stakeholders to simulate "what-if" scenarios
- Level the playing field between institutional and retail creditors

## Citation

```bibtex
@article{verma2026ibc,
  title={Insolvency Without Predictability? A Computational Census of 4,820 IBC Cases and the Information Asymmetry Problem},
  author={Verma, Prakhar},
  year={2026},
  note={Indian Institute of Management Rohtak; Indian Institute of Technology Madras}
}
```

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE)

## Author

**Prakhar Verma**  
Indian Institute of Management Rohtak (BBA-LLB)  
Indian Institute of Technology Madras (BS Data Science)
