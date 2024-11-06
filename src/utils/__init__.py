"""
Utility modules for image inpainting operations including data loading,
image processing, mask generation, and model weight management.
"""

from typing import List, Dict, Any, Optional, Tuple

# Core processing imports
from .image_processor import ImageProcessor, ProcessingConfig
from .mask_generator import MaskGenerator, MaskConfig
from .data_loader import InpaintingDataset, get_data_loader
from .weight_loader import WeightLoader

# Test data management imports
from .organize_test_data import setup_test_directories, organize_test_files, verify_test_files
from .rename_test_files import TestFileOrganizer
from .manage_test_data import setup_test_environment

# Version information
__version__ = '1.0.0'

__all__ = [
    # Core components
    'ImageProcessor',
    'ProcessingConfig',
    'MaskGenerator',
    'MaskConfig',
    'InpaintingDataset',
    'get_data_loader',
    'WeightLoader',
    
    # Test data organization
    'setup_test_directories',
    'organize_test_files',
    'verify_test_files',
    'TestFileOrganizer',
    'setup_test_environment'
]

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())