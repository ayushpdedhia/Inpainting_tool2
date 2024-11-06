"""
Core module initialization for image inpainting application.

This module provides access to the main components needed for image inpainting:
- ModelManager: Handles model operations and inference
- MaskGenerator: Generates masks for inpainting
- PConvUNet: Implementation of Partial Convolution-based UNet
- Additional utilities for model evaluation and data processing
"""

from .model_manager import ModelManager
from src.utils.mask_generator import MaskGenerator
from src.models.pconv.models.pconv_unet import PConvUNet
#from src.utils.metrics import PSNR, SSIM, InceptionScore
from src.utils.data_loader import InpaintingDataset, get_data_loader
from src.models.pconv.loss import PConvLoss

__version__ = '1.0.0'

__all__ = [
    # Core components
    'ModelManager',    # Main class for managing model operations
    'MaskGenerator',   # Class for generating masks
    'PConvUNet',      # The partial convolution UNet architecture
    
    # Evaluation metrics
    #'PSNR',           # Peak Signal-to-Noise Ratio metric
    #'SSIM',           # Structural Similarity Index metric
    #'InceptionScore', # Inception Score metric
    
    # Data handling
    'InpaintingDataset',  # Dataset class for inpainting
    'get_data_loader',    # Function to create data loaders
    
    # Loss functions
    'PConvLoss',      # Partial Convolution loss function
]

# Configure logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())