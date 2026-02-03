"""
Configuration management for IBC Prediction project.
Loads configuration from environment variables and .env file.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)


@dataclass
class DataConfig:
    """Data paths configuration."""
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('DATA_DIR', './data')))
    extracted_json_path: Path = field(default_factory=lambda: Path(os.getenv('EXTRACTED_JSON_PATH', './data/extracted_cases.json')))
    raw_data_path: Path = field(default_factory=lambda: Path(os.getenv('RAW_DATA_PATH', './data/nclt_judgments')))
    ocr_output_path: Path = field(default_factory=lambda: Path(os.getenv('OCR_OUTPUT_PATH', './data/ocroutput')))
    graph_data_path: Path = field(default_factory=lambda: Path(os.getenv('GRAPH_DATA_PATH', './data/ibc_graphs.pt')))
    physarum_data_path: Path = field(default_factory=lambda: Path(os.getenv('PHYSARUM_DATA_PATH', './data/physarum_dataset_new_big.jsonl')))
    
    def __post_init__(self):
        # Ensure all directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.ocr_output_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_dim: int = int(os.getenv('HIDDEN_DIM', 128))
    num_gcn_layers: int = int(os.getenv('NUM_GCN_LAYERS', 3))
    dropout: float = float(os.getenv('DROPOUT', 0.4))
    num_classes: int = int(os.getenv('NUM_CLASSES', 3))
    input_dim_physarum: int = 3
    input_dim_ibc: int = 22


@dataclass
class TrainingConfig:
    """Training configuration."""
    seed: int = int(os.getenv('SEED', 42))
    device: str = os.getenv('DEVICE', 'cuda')
    
    # Phase 1
    phase1_epochs: int = int(os.getenv('PHASE1_EPOCHS', 50))
    phase1_lr: float = float(os.getenv('PHASE1_LR', 0.001))
    phase1_batch_size: int = int(os.getenv('PHASE1_BATCH_SIZE', 32))
    
    # Phase 2
    phase2_epochs: int = int(os.getenv('PHASE2_EPOCHS', 100))
    phase2_lr: float = float(os.getenv('PHASE2_LR', 0.0005))
    phase2_batch_size: int = int(os.getenv('PHASE2_BATCH_SIZE', 32))
    
    # Optimization
    weight_decay: float = float(os.getenv('WEIGHT_DECAY', 0.0001))
    grad_clip: float = float(os.getenv('GRAD_CLIP', 1.0))
    
    # Loss weights
    edge_loss_weight: float = float(os.getenv('EDGE_LOSS_WEIGHT', 0.3))
    class_loss_weight: float = float(os.getenv('CLASS_LOSS_WEIGHT', 1.0))


@dataclass
class InferenceConfig:
    """Inference configuration."""
    default_case_index: int = int(os.getenv('DEFAULT_CASE_INDEX', 256))
    mc_samples: int = int(os.getenv('MC_SAMPLES', 5000))
    mc_seed: int = int(os.getenv('MC_SEED', 42))
    class_names: List[str] = field(default_factory=lambda: ["Strategic", "Promoter", "Liquidation"])


@dataclass
class PathsConfig:
    """Model paths configuration."""
    output_dir: Path = field(default_factory=lambda: Path(os.getenv('MODEL_OUTPUT_DIR', './outputs')))
    pretrained_model_path: Path = field(default_factory=lambda: Path(os.getenv('PRETRAINED_MODEL_PATH', './outputs/pretrained_dim3.pt')))
    final_model_path: Path = field(default_factory=lambda: Path(os.getenv('FINAL_MODEL_PATH', './outputs/final_model.safetensors')))
    final_model_pt_path: Path = field(default_factory=lambda: Path(os.getenv('FINAL_MODEL_PT_PATH', './outputs/final_model.pt')))
    figures_dir: Path = field(default_factory=lambda: Path(os.getenv('MODEL_OUTPUT_DIR', './outputs')) / 'figures')
    mc_output_dir: Path = field(default_factory=lambda: Path('./outputs/mc'))
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.mc_output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class APIConfig:
    """API keys configuration."""
    openrouter_api_key: Optional[str] = field(default_factory=lambda: os.getenv('OPENROUTER_API_KEY'))
    openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions"
    
    def validate(self) -> bool:
        """Check if required API keys are set."""
        return self.openrouter_api_key is not None and len(self.openrouter_api_key) > 0


@dataclass
class OCRConfig:
    """OCR pipeline configuration."""
    num_feeder_workers: int = int(os.getenv('NUM_FEEDER_WORKERS', 4))
    batch_size: int = int(os.getenv('OCR_BATCH_SIZE', 16))
    buffer_size: int = 64
    dpi: int = int(os.getenv('OCR_DPI', 300))
    jpeg_quality: int = int(os.getenv('JPEG_QUALITY', 95))
    use_fp16: bool = True
    page_timeout: float = 60.0
    threads_per_worker: int = 3


@dataclass
class ScraperConfig:
    """Web scraper configuration."""
    base_url: str = os.getenv('IBBI_BASE_URL', 'https://ibbi.gov.in/orders/nclt')
    start_page: int = int(os.getenv('START_PAGE', 689))
    end_page: int = int(os.getenv('END_PAGE', 1478))
    download_folder: Path = field(default_factory=lambda: Path(os.getenv('DOWNLOAD_FOLDER', './data/nclt_judgments')))
    target_keywords: List[str] = field(default_factory=lambda: [
        "Final Order", "Resolution Plan", "Liquidation", "Section 12A"
    ])


# Global configuration instances
data_config = DataConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
inference_config = InferenceConfig()
paths_config = PathsConfig()
api_config = APIConfig()
ocr_config = OCRConfig()
scraper_config = ScraperConfig()


# Node types for graph construction
NODE_TYPES = {
    'CASE_START': 0,
    'FC_PSU_BANK': 1, 'FC_PRIVATE_BANK': 2, 'FC_NBFC': 3, 'FC_ARC': 4, 'FC_OTHER': 5,
    'OC_POOL': 6,
    'COC_ALIGNED': 7, 'COC_FRAGMENTED': 8,
    'PROMOTER_COOPERATIVE': 9, 'PROMOTER_HOSTILE': 10,
    'PROMOTER_29A_ELIGIBLE': 11, 'PROMOTER_29A_BLOCKED': 12,
    'TIMELINE_NORMAL': 13, 'TIMELINE_EXTENDED': 14,
    'RESOLUTION_STRATEGIC': 15, 'RESOLUTION_PROMOTER': 16, 'LIQUIDATION': 17,
}
