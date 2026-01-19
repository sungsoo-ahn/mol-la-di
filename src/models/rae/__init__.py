"""RAE (Representation Autoencoder) module for molecule generation.

RAE uses a frozen pretrained encoder (MAE) with a trainable decoder
that includes noise augmentation for robustness to diffusion outputs.

Reference: Diffusion Transformers with Representation Autoencoders
           https://arxiv.org/abs/2510.11690
"""

from src.models.rae.encoder_adapter import MAEEncoderAdapter
from src.models.rae.decoder import RAEDecoder
from src.models.rae.loss import (
    compute_rae_loss,
    add_training_noise,
    FocalLoss,
    get_noise_sigma,
    compute_edge_class_weights,
)
from src.models.rae.rae_model import RAEModel

__all__ = [
    'MAEEncoderAdapter',
    'RAEDecoder',
    'RAEModel',
    'compute_rae_loss',
    'add_training_noise',
    'FocalLoss',
    'get_noise_sigma',
    'compute_edge_class_weights',
]
