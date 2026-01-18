"""Utility functions for the molecule generation project."""

import random
import numpy as np
import torch
import yaml
from pathlib import Path


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, path: str) -> None:
    """Save configuration to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_output_dir(config: dict) -> Path:
    """Create output directory structure from config."""
    output_dir = Path(config['output_dir'])
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    return output_dir


def get_device(config: dict) -> torch.device:
    """Get device from config or default to cuda if available."""
    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adj_to_dense(edge_index: torch.Tensor, num_nodes: int, edge_attr: torch.Tensor = None) -> torch.Tensor:
    """Convert sparse edge_index to dense adjacency matrix."""
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    if edge_attr is not None:
        adj[edge_index[0], edge_index[1]] = edge_attr.float()
    else:
        adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def dense_to_edge_index(adj: torch.Tensor) -> tuple:
    """Convert dense adjacency matrix to edge_index format."""
    edge_index = torch.nonzero(adj, as_tuple=False).t().contiguous()
    edge_attr = adj[edge_index[0], edge_index[1]]
    return edge_index, edge_attr
