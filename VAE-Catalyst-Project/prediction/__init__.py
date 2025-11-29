"""
Inference Package

This package contains prediction and latent space exploration
scripts for the VAE Catalyst Project.
"""

from .predict_single import load_model, predict, main as predict_main
from .predict_latent_exploration import (
    traverse_latent_space_and_predict, 
    save_results_to_excel, 
    main as explore_main
)

__all__ = [
    'load_model',
    'predict',
    'predict_main',
    'traverse_latent_space_and_predict',
    'save_results_to_excel', 
    'explore_main',
]

__version__ = '1.0.0'
