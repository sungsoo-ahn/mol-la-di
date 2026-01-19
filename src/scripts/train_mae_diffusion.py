"""Training script for MAE-based latent diffusion model.

This is a thin wrapper around train_latent_diffusion.py for backward compatibility.
The config should specify mae.checkpoint for MAE mode.
"""

import sys
from src.scripts.train_latent_diffusion import main


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/mae_diffusion/qm9_mae_diffusion.yaml"
    main(config_path)
