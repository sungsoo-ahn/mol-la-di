"""Diffusion models for latent space generation."""

import warnings

from src.models.diffusion.noise_scheduler import DDPMScheduler
from src.models.diffusion.dit_block import DiTBlock
from src.models.diffusion.latent_diffusion import (
    LatentDiffusionModel,
    LatentDiffusionWithRAE,
    LatentDiffusionWithMAE,
)


def LatentDiffusionForMAE(*args, **kwargs):
    """Deprecated: Use LatentDiffusionModel(latent_mode='nodes_and_edges') instead."""
    warnings.warn(
        "LatentDiffusionForMAE is deprecated. Use LatentDiffusionModel(latent_mode='nodes_and_edges') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs['latent_mode'] = 'nodes_and_edges'
    return LatentDiffusionModel(*args, **kwargs)


__all__ = [
    "DDPMScheduler",
    "DiTBlock",
    "LatentDiffusionModel",
    "LatentDiffusionWithRAE",
    "LatentDiffusionWithMAE",
    "LatentDiffusionForMAE",  # Deprecated alias for backward compatibility
]
