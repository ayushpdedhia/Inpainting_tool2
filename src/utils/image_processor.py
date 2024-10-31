# src/utils/image_processor.py

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Union, Optional
import logging
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters"""
    target_size: Tuple[int, int] = (512, 512)
    normalize_range: Tuple[float, float] = (0.0, 1.0)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet means
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)   # ImageNet stds

class ImageProcessor:
    """Handles image preprocessing and postprocessing operations for the inpainting model"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the image processor with given configuration"""
        self.config = config or ProcessingConfig()
        logger.info(f"Initialized ImageProcessor with target size: {self.config.target_size}")

    def preprocess(self, 
                  image: Union[Image.Image, np.ndarray], 
                  mask: np.ndarray,
                  target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image and mask for model input.
        
        Args:
            image: Input image (PIL Image or numpy array)
            mask: Input mask (numpy array)
            target_size: Optional target size, overrides config if provided
            
        Returns:
            Tuple of preprocessed (image, mask) as numpy arrays
        """
        try:
            # Convert PIL Image to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)

            # Resize inputs
            size = target_size or self.config.target_size
            processed_image = self._resize_image(image, size)
            processed_mask = self._resize_mask(mask, size)

            # Normalize image
            processed_image = self._normalize_image(processed_image)
            
            # Normalize mask to binary
            processed_mask = (processed_mask > 127.5).astype(np.float32)

            # Add batch dimension if needed
            if len(processed_image.shape) == 3:
                processed_image = np.expand_dims(processed_image, 0)
            if len(processed_mask.shape) == 2:
                processed_mask = np.expand_dims(processed_mask, [0, -1])

            return processed_image, processed_mask

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise RuntimeError(f"Failed to preprocess inputs: {str(e)}")

    def _resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image maintaining aspect ratio"""
        if image.shape[:2] != size:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return image

    def _resize_mask(self, mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize mask maintaining binary values"""
        if mask.shape[:2] != size:
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        return mask

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image values"""
        # Scale to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize using ImageNet stats if configured
        if self.config.mean and self.config.std:
            image = (image - self.config.mean) / self.config.std
            
        return image

    def postprocess(self, image: np.ndarray) -> Union[Image.Image, np.ndarray]:
        """
        Postprocess model output back to displayable image.
        
        Args:
            image: Model output as numpy array
            
        Returns:
            Processed image as PIL Image or numpy array
        """
        try:
            # Remove batch dimension if present
            if image.ndim == 4:
                image = image[0]
            
            # Denormalize
            if self.config.mean and self.config.std:
                image = (image * self.config.std) + self.config.mean
                
            # Clip values to valid range
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            
            return Image.fromarray(image)

        except Exception as e:
            logger.error(f"Error in postprocessing: {str(e)}")
            raise RuntimeError(f"Failed to postprocess output: {str(e)}")

    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor"""
        # Handle the case where image is already a tensor
        if isinstance(image, torch.Tensor):
            return image
            
        # Convert HWC to CHW format and normalize
        if image.ndim == 3:
            image = image.transpose(2, 0, 1)
        
        return torch.from_numpy(image).float()

    def from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array"""
        # Move to CPU if needed
        output = tensor.cpu().numpy()
        
        # Convert CHW back to HWC format
        if output.ndim == 3:
            output = output.transpose(1, 2, 0)
            
        return output