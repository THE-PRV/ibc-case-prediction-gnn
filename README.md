# IBC Prediction Project

A deep learning project for predicting outcomes of Insolvency and Bankruptcy Code (IBC) cases in India using Graph Convolutional Networks (GCNs).

## Overview

This project uses a Physarum-inspired Graph Convolutional Network to predict three possible outcomes of IBC cases:
- **Strategic Resolution** (Class 0)
- **Promoter Resolution** (Class 1)
- **Liquidation** (Class 2)

## Project Structure

```
.
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── data/              # Data processing utilities
│   ├── training/          # Training scripts
│   ├── inference/         # Inference and prediction
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── data/                  # Data directory (not tracked by git)
├── models/                # Saved models (not tracked by git)
├── outputs/               # Training outputs and figures
├── scripts/               # Helper scripts
├── notebooks/             # Jupyter notebooks (if any)
└── tests/                 # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ibcprediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
```bash
cp config/.env.example config/.env
```

2. Edit `config/.env` with your paths and API keys:
```bash
# Data paths
DATA_DIR=/path/to/your/data
RAW_DATA_PATH=/path/to/raw/data
EXTRACTED_JSON_PATH=/path/to/extracted/json

# Model paths
MODEL_OUTPUT_DIR=./models
PRETRAINED_MODEL_PATH=./models/pretrained_dim3.pt
FINAL_MODEL_PATH=./models/final_model.safetensors

# API Keys (if using extraction features)
OPENROUTER_API_KEY=your_api_key_here
```

## Usage

### 1. Data Preparation

Convert extracted JSON data to graph format:
```bash
python src/data/dataconverter.py --input data/extracted_cases.json --output data/ibc_graphs.pt
```

### 2. Training

Train the model with the default configuration:
```bash
python src/training/train_physarum_gcn.py
```

Or with custom config:
```bash
python src/training/train_physarum_gcn.py --config config/training_config.yaml
```

### 3. Generate Figures

Create publication-ready figures:
```bash
python src/utils/generate_figures.py
```

### 4. Inference

Run what-if analysis on a specific case:
```bash
python src/inference/whatifn.py --case-index 256 --mc 5000 --outdir outputs/mc
```

## Model Architecture

The model uses a Physarum-inspired GCN with the following components:
- Node encoder with dropout
- 3-layer GCN with residual connections and LayerNorm
- Edge predictor for graph reconstruction
- Graph-level classifier using mean and max pooling

## Data Pipeline

1. **Scraping** (`src/data/scraper.py`): Downloads NCLT judgments from IBBI website
2. **OCR** (`src/data/ocr_pipeline.py`): Extracts text from PDFs using PaddleOCR
3. **Extraction** (`src/data/ibc_extractor.py`): Extracts structured data using LLMs
4. **Conversion** (`src/data/dataconverter.py`): Converts JSON to PyTorch Geometric graphs

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ibcprediction2025,
  title={IBC Outcome Prediction using Physarum-Inspired Graph Neural Networks},
  author={Prakhar Verma 
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch Geometric for the GCN implementation
- PaddleOCR for text extraction
- OpenRouter for LLM API access
