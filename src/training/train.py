"""
Physarum-inspired GCN Training Script

Trains the model in two phases:
1. Pretraining on Physarum data (3 features)
2. Fine-tuning on IBC data (22 features)

Features:
- WeightedRandomSampler for class imbalance
- Automatic mixed precision training
- Gradient clipping
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import WeightedRandomSampler
import numpy as np
import json
from pathlib import Path
from typing import List, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from ..models.physarum_gcn import PhysarumGCN
from ..utils.config import (
    training_config, model_config, paths_config, data_config
)


# Safetensors support
try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: 'safetensors' not found. Install with: pip install safetensors")


class PhysarumLoss(nn.Module):
    """
    Combined loss for edge prediction and node classification.
    """
    
    def __init__(self, edge_weight: float = 0.3, class_weight: float = 1.0):
        super().__init__()
        self.edge_weight = edge_weight
        self.class_weight = class_weight
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(
        self,
        edge_pred: torch.Tensor,
        edge_target: torch.Tensor,
        logits: torch.Tensor,
        class_target: torch.Tensor,
        mode: str = 'both'
    ) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            edge_pred: Predicted edge strengths
            edge_target: Target edge strengths
            logits: Classification logits
            class_target: Target classes
            mode: 'edge', 'outcome', or 'both'
        
        Returns:
            Combined loss
        """
        loss = 0.0
        
        if mode in ['edge', 'both'] and edge_target is not None:
            loss += self.edge_weight * self.bce(edge_pred, edge_target)
        
        if mode in ['outcome', 'both']:
            loss += self.class_weight * self.ce(logits, class_target)
        
        return loss


def physarum_graph_to_pyg(graph: dict) -> Optional[Data]:
    """
    Convert raw Physarum JSON dict to PyG Data object (3 features).
    
    Args:
        graph: Physarum graph dictionary
    
    Returns:
        PyG Data object or None if conversion fails
    """
    try:
        nodes = graph["nodes"]
        edges = graph["edges"]
        
        # Features (3 dimensions)
        node_feats = []
        for n in nodes:
            # [log_adhesion, particle_density, sensitivity]
            feats = [
                np.log1p(n.get("adhesion_strength", 0.0)),
                n.get("particle_density", 0.5),
                n.get("path_sensitivity", 0.5)
            ]
            node_feats.append(feats)
        
        x = torch.tensor(node_feats, dtype=torch.float)
        
        # Edges
        edge_index = torch.tensor(
            [[e["source"], e["target"]] for e in edges],
            dtype=torch.long
        ).t().contiguous()
        
        # Edge attributes (strength)
        edge_attr = torch.tensor(
            [[e.get("strength", 0.5)] for e in edges],
            dtype=torch.float
        )
        
        # Edge targets for reconstruction
        y = torch.tensor(
            [e.get("strength", 0.5) for e in edges],
            dtype=torch.float
        )
        
        # Dummy outcome for compatibility
        outcome = torch.tensor([0], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, outcome=outcome)
        
    except Exception as e:
        print(f"Failed to convert Physarum graph: {e}")
        return None


def load_physarum_from_jsonl(file_path: Path) -> List[Data]:
    """
    Load Physarum graphs from JSONL file.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of PyG Data objects
    """
    if not file_path.exists():
        print(f"Physarum file not found: {file_path}")
        return []
    
    graphs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    graph_dict = json.loads(line)
                    data = physarum_graph_to_pyg(graph_dict)
                    if data is not None:
                        graphs.append(data)
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(graphs)} Physarum graphs from {file_path}")
    return graphs


def load_ibc_cases(file_path: Path) -> List[Data]:
    """
    Load IBC cases from .pt file.
    
    Args:
        file_path: Path to .pt file
    
    Returns:
        List of PyG Data objects
    """
    if not file_path.exists():
        raise FileNotFoundError(f"IBC data file not found: {file_path}")
    
    # Load with weights_only=True for security
    graphs = torch.load(file_path, weights_only=True)
    print(f"Loaded {len(graphs)} IBC cases from {file_path}")
    return graphs


def train_epoch(
    model: PhysarumGCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: PhysarumLoss,
    device: str,
    mode: str = 'both'
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        loader: DataLoader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        mode: Training mode ('edge', 'outcome', or 'both')
    
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        logits, edge_strengths = model(batch)
        
        # Safe target retrieval
        edge_target = getattr(batch, 'y', None)
        if edge_target is not None:
            edge_target = edge_target.to(device)
        
        # Squeeze with explicit dimension to avoid issues
        class_target = batch.outcome.squeeze(-1) if batch.outcome.dim() > 1 else batch.outcome
        
        loss = loss_fn(edge_strengths, edge_target, logits, class_target, mode)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: PhysarumGCN,
    loader: DataLoader,
    device: str
) -> tuple[float, list, list]:
    """
    Evaluate model.
    
    Args:
        model: Model to evaluate
        loader: DataLoader
        device: Device to evaluate on
    
    Returns:
        Tuple of (F1 score, predictions, targets)
    """
    model.eval()
    preds, targets = [], []
    
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        targets.extend(batch.outcome.cpu().numpy())
    
    f1 = f1_score(targets, preds, average='weighted', zero_division=0)
    return f1, preds, targets


def main():
    """Main training function."""
    print("=" * 70)
    print("PHYSARUM-GCN: HYBRID TRAINING WITH OVERSAMPLING")
    print("=" * 70)
    
    # Set random seeds
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    
    # Determine device
    device = training_config.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Phase 1: Pretraining (3 features)
    physarum_graphs = load_physarum_from_jsonl(data_config.physarum_data_path)
    
    if physarum_graphs:
        print(f"\n{'='*70}")
        print("PHASE 1: PRETRAINING (Dim=3)")
        print(f"{'='*70}")
        
        loader = DataLoader(
            physarum_graphs,
            batch_size=training_config.phase1_batch_size,
            shuffle=True
        )
        
        model = PhysarumGCN(
            input_dim=model_config.input_dim_physarum,
            hidden_dim=model_config.hidden_dim,
            num_classes=model_config.num_classes,
            num_gcn_layers=model_config.num_gcn_layers,
            dropout=model_config.dropout
        ).to(device)
        
        optimizer = AdamW(model.parameters(), lr=training_config.phase1_lr)
        loss_fn = PhysarumLoss(
            edge_weight=training_config.edge_loss_weight,
            class_weight=training_config.class_loss_weight
        ).to(device)
        
        for epoch in range(1, training_config.phase1_epochs + 1):
            loss = train_epoch(model, loader, optimizer, loss_fn, device, mode='edge')
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{training_config.phase1_epochs}: Loss={loss:.4f}")
        
        # Save pretrained model
        save_path = paths_config.pretrained_model_path
        torch.save(model.state_dict(), save_path)
        print(f"✓ Pretrained model saved to {save_path}")
    else:
        print("⚠ Skipping Phase 1 (No Physarum data found)")
        model = None
    
    # Phase 2: Fine-tuning (22 features)
    print(f"\n{'='*70}")
    print("PHASE 2: FINE-TUNING (Dim=22)")
    print(f"{'='*70}")
    
    ibc_graphs = load_ibc_cases(data_config.graph_data_path)
    ibc_dim = ibc_graphs[0].x.shape[1]
    print(f"IBC input dimension: {ibc_dim}")
    
    if model is None:
        # Initialize fresh model
        model = PhysarumGCN(
            input_dim=ibc_dim,
            hidden_dim=model_config.hidden_dim,
            num_classes=model_config.num_classes,
            num_gcn_layers=model_config.num_gcn_layers,
            dropout=model_config.dropout
        ).to(device)
    else:
        # Swap input layer for IBC dimension
        print(f"Swapping input layer: {model_config.input_dim_physarum} -> {ibc_dim}")
        model.node_encoder[0] = nn.Linear(ibc_dim, model_config.hidden_dim).to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.phase2_lr,
        weight_decay=training_config.weight_decay
    )
    
    loss_fn = PhysarumLoss(
        edge_weight=training_config.edge_loss_weight,
        class_weight=training_config.class_loss_weight
    ).to(device)
    
    # Stratified split
    train_g, test_g = train_test_split(
        ibc_graphs,
        test_size=0.2,
        stratify=[g.outcome.item() for g in ibc_graphs],
        random_state=training_config.seed
    )
    
    print(f"Train: {len(train_g)}, Test: {len(test_g)}")
    
    # WeightedRandomSampler for class balance
    print("Configuring WeightedRandomSampler for class balance...")
    train_targets = [g.outcome.item() for g in train_g]
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(
        train_g,
        batch_size=training_config.phase2_batch_size,
        sampler=sampler
    )
    test_loader = DataLoader(test_g, batch_size=training_config.phase2_batch_size)
    
    # Training loop
    best_f1 = 0.0
    best_epoch = 0
    history = []
    
    for epoch in range(1, training_config.phase2_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device, mode='outcome')
        val_f1, _, _ = evaluate(model, test_loader, device)
        
        history.append({
            'epoch': epoch,
            'loss': loss,
            'val_f1': val_f1
        })
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            
            # Save best model
            torch.save(model.state_dict(), paths_config.final_model_pt_path)
            
            # Also save as safetensors if available
            if SAFETENSORS_AVAILABLE:
                save_file(
                    model.state_dict(),
                    paths_config.final_model_path
                )
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{training_config.phase2_epochs}: "
                  f"Loss={loss:.4f}, Val F1={val_f1:.4f} (Best: {best_f1:.4f} @ {best_epoch})")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")
    
    # Load best model
    model.load_state_dict(
        torch.load(paths_config.final_model_pt_path, map_location=device, weights_only=True)
    )
    
    test_f1, preds, targets = evaluate(model, test_loader, device)
    
    print(f"\nBest model from epoch {best_epoch}")
    print(f"Test F1: {test_f1:.4f}\n")
    
    print(classification_report(
        targets,
        preds,
        target_names=training_config.class_names
    ))
    
    # Save history
    history_path = paths_config.output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training history saved to {history_path}")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
