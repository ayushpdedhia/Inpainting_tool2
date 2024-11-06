# D:\Inpainting_tool2\src\models\__init__.py
"""
Model Registry and Management System for Inpainting Models
"""
from typing import Dict, Type, Optional, Any, List, Tuple
from abc import ABC, abstractmethod
import torch.nn as nn
import logging
from packaging import version
import torch

logger = logging.getLogger(__name__)

class InpaintingModel(ABC):
    """Abstract base class for all inpainting models"""
    
    @abstractmethod
    def forward(self, image, mask):
        """Forward pass of the model"""
        pass
    
    @property
    @abstractmethod
    def model_version(self) -> str:
        """Return the model version"""
        pass

    @property
    @abstractmethod
    def required_inputs(self) -> Dict[str, Any]:
        """Return required input specifications"""
        pass
    
    @abstractmethod
    def get_model_size(self) -> Tuple[int, int]:
        """Return model size (parameters, buffers)"""
        pass

class ModelRegistry:
    """
    Model Registry implementing Factory Pattern for Inpainting Models
    """
    _models: Dict[str, Type[InpaintingModel]] = {}
    _versions: Dict[str, str] = {}
    _compatibility: Dict[str, Dict[str, Any]] = {}
    _descriptions: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, version: str, description: str = "", **compatibility):
        """
        Decorator to register a model class
        
        Args:
            name: Unique identifier for the model
            version: Model version string
            description: Short description of the model
            **compatibility: Compatibility requirements (torch_version, cuda_version, etc.)
        """
        def wrapper(model_cls: Type[InpaintingModel]):
            if name in cls._models:
                logger.warning(f"Model {name} is already registered. Overwriting...")
            
            cls._models[name] = model_cls
            cls._versions[name] = version
            cls._compatibility[name] = compatibility
            cls._descriptions[name] = description
            logger.info(f"Successfully registered model: {name} (v{version})")
            return model_cls
        return wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[InpaintingModel]:
        """
        Create an instance of a registered model
        
        Args:
            name: Model identifier
            **kwargs: Model initialization parameters
            
        Returns:
            Instance of the requested model or None if not found
        """
        if name not in cls._models:
            available_models = ", ".join(cls._models.keys())
            logger.error(f"Model {name} not found. Available models: {available_models}")
            return None
            
        try:
            # Check compatibility
            cls._check_compatibility(name)
            
            # Create model instance
            model = cls._models[name](**kwargs)
            logger.info(f"Successfully created model: {name} (v{cls._versions[name]})")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model {name}: {str(e)}")
            raise

    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        """Return detailed information about registered models"""
        return {
            name: {
                "version": cls._versions[name],
                "description": cls._descriptions[name],
                "compatibility": cls._compatibility[name]
            }
            for name in cls._models
        }

    @classmethod
    def get_compatibility(cls, name: str) -> Dict[str, Any]:
        """Return compatibility requirements for a model"""
        if name not in cls._compatibility:
            raise KeyError(f"Model {name} not found in registry")
        return cls._compatibility[name].copy()

    @classmethod
    def _check_compatibility(cls, name: str):
        """Check if the current environment meets model compatibility requirements"""
        compat = cls._compatibility.get(name, {})
        
        # Check PyTorch version if specified
        if 'torch_version' in compat:
            required = version.parse(compat['torch_version'])
            current = version.parse(torch.__version__.split('+')[0])
            if current < required:
                raise RuntimeError(
                    f"Model {name} requires PyTorch >= {required}, "
                    f"but found version {current}"
                )

        # Check CUDA version if specified
        if 'cuda_version' in compat:
            if not torch.cuda.is_available():
                raise RuntimeError(f"Model {name} requires CUDA but no CUDA device is available")
            
            current_cuda = torch.version.cuda
            required_cuda = compat['cuda_version']
            if version.parse(current_cuda) < version.parse(required_cuda):
                raise RuntimeError(
                    f"Model {name} requires CUDA >= {required_cuda}, "
                    f"but found version {current_cuda}"
                )

def register_all_models():
    """Register all available models"""
    from .pconv import PConvUNet
    #from .other_models import OtherInpaintingModels  # Import other models when available
    
    # Register PConv model
    ModelRegistry.register(
        name="pconv_unet",
        version="1.0.0",
        description="Partial Convolution based UNet for image inpainting",
        torch_version="2.0.0",
        cuda_version="11.7",
        min_memory_gb=6  # Added minimum memory requirement
    )(PConvUNet)
    
    # Register other models here as they become available

# Export important classes and functions
__all__ = [
    'InpaintingModel',
    'ModelRegistry',
    #'register_all_models'
]

# Register all models when the package is imported
register_all_models()