"""
Physarum-inspired Graph Convolutional Network for IBC Outcome Prediction.

This module contains the GCN model architecture with edge prediction capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class EdgePredictor(nn.Module):
    """MLP for predicting edge strengths between nodes."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, 1]
        
        Returns:
            Edge predictions [num_edges]
        """
        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst], edge_attr], dim=1)
        return self.mlp(edge_features).squeeze(-1)


class PhysarumGCN(nn.Module):
    """
    Physarum-inspired GCN for IBC outcome prediction.
    
    Architecture:
    - Node encoder (MLP with dropout)
    - 3 GCN layers with residual connections and LayerNorm
    - Edge predictor for graph reconstruction
    - Graph-level classifier (mean + max pooling)
    """
    
    def __init__(
        self,
        input_dim: int = 22,
        hidden_dim: int = 128,
        num_classes: int = 3,
        num_gcn_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GCN layers with residual connections
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_gcn_layers)
        ])
        
        # Layer normalization for each GCN layer
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gcn_layers)
        ])
        
        # Edge predictor
        self.edge_predictor = EdgePredictor(hidden_dim, dropout)
        
        # Graph-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Encode nodes through GCN layers.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, 1] (optional)
        
        Returns:
            Encoded node features [num_nodes, hidden_dim]
        """
        h = self.node_encoder(x)
        
        # Extract edge weights if available
        edge_weight = edge_attr.squeeze(-1) if edge_attr is not None else None
        
        # Apply GCN layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_weight=edge_weight)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # Residual connection
        
        return h
    
    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
            edge_strengths: Edge predictions [num_edges]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Create batch vector if not present
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Encode nodes
        h = self.encode(x, edge_index, edge_attr)
        
        # Predict edge strengths
        edge_strengths = self.edge_predictor(h, edge_index, edge_attr)
        
        # Graph-level pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=1)
        
        # Classification
        logits = self.classifier(h_graph)
        
        return logits, edge_strengths
    
    def predict(self, data) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            data: PyG Data object
        
        Returns:
            Class probabilities [batch_size, num_classes]
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(data)
            probs = F.softmax(logits, dim=1)
        return probs


def load_model(
    model_path: str,
    input_dim: int = 22,
    hidden_dim: int = 128,
    num_classes: int = 3,
    device: str = 'cuda'
) -> PhysarumGCN:
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to model file (.pt or .safetensors)
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_classes: Number of output classes
        device: Device to load model on
    
    Returns:
        Loaded PhysarumGCN model
    """
    import os
    from pathlib import Path
    
    model = PhysarumGCN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    
    # Move model to device FIRST (fixes device placement bug)
    model = model.to(device)
    
    path = Path(model_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load weights based on file extension
    if path.suffix == '.safetensors':
        try:
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
            model.load_state_dict(state_dict)
        except ImportError:
            raise ImportError("safetensors library not found. Install with: pip install safetensors")
    else:
        # Use weights_only=True for security (PyTorch 2.0+)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    
    model.eval()
    return model


def save_model(
    model: PhysarumGCN,
    output_path: str,
    use_safetensors: bool = True
) -> None:
    """
    Save model to file.
    
    Args:
        model: Model to save
        output_path: Output file path
        use_safetensors: Whether to use safetensors format
    """
    from pathlib import Path
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if use_safetensors and path.suffix == '.safetensors':
        try:
            from safetensors.torch import save_file
            save_file(model.state_dict(), output_path)
        except ImportError:
            # Fall back to torch.save
            torch.save(model.state_dict(), str(path.with_suffix('.pt')))
    else:
        torch.save(model.state_dict(), output_path)
