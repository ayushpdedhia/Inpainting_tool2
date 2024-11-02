"""
Inpainting Tool Package
----------------------
Core functionality for image inpainting using various deep learning models.
"""

# Version info
__version__ = '1.0.0'

try:
    # Core functionality
    from .core.model_manager import ModelManager
    
    # Interface components
    from .interface.app import InpaintingApp, main
    
    # Model architectures
    from .models.pconv.models.pconv_unet import PConvUNet
    
    # Utilities
    from .utils.image_processor import ImageProcessor
    from .utils.mask_generator import MaskGenerator
    
    __all__ = [
        # Core
        'ModelManager',
        
        # Interface
        'InpaintingApp',
        'main',
        
        # Models
        'PConvUNet',
        
        # Utils
        'ImageProcessor',
        'MaskGenerator'
    ]

except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(
        f"Some imports failed: {str(e)}. This is expected during test environment setup."
    )
    
    __all__ = []

# Package metadata
__author__ = 'Your Name'
__description__ = 'Image Inpainting Tool using Deep Learning'