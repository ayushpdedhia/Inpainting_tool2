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
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        logger.info(f"Initialized ImageProcessor with target size: {self.config.target_size}")


    def preprocess(self, 
                image: Union[Image.Image, np.ndarray], 
                mask: np.ndarray,
                target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Handle batch inputs
            is_batch = len(image.shape) == 4
            if is_batch:
                return self._preprocess_batch(image, mask, target_size)
            
            # Validate input
            if len(image.shape) == 3 and image.shape[2] == 4:
                raise ValueError("4-channel images are not supported")
            
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError("Image and mask dimensions must match")

            size = target_size or self.config.target_size
            if not size or len(size) != 2 or size[0] <= 0 or size[1] <= 0:
                raise ValueError("Invalid target size")

            # Process image to (C, H, W) format
            processed_image = self._resize_image(image, size)
            processed_image = self._normalize_image(processed_image)
            
            # Process mask to (1, H, W) format
            processed_mask = self._resize_mask(mask, size)
            processed_mask = (processed_mask > 127.5).astype(np.float32)
            if len(processed_mask.shape) == 2:
                processed_mask = np.expand_dims(processed_mask, 0)
            
            # Add batch dimension to make (B, C, H, W)
            processed_image = np.expand_dims(processed_image, 0)
            processed_mask = np.expand_dims(processed_mask, 0)
            
            # Now shape should be (B, C, H, W)
            assert processed_image.shape[-2:] == size  # Height, Width should match target size
            assert processed_mask.shape[-2:] == size
            
            return processed_image, processed_mask

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise RuntimeError(f"Failed to preprocess inputs: {str(e)}")

    def _preprocess_batch(self, 
                        images: np.ndarray, 
                        masks: np.ndarray,
                        target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Handle batch preprocessing with channels-first format"""
        size = target_size or self.config.target_size
        batch_size = images.shape[0]
        
        processed_images = []
        processed_masks = []
        
        for i in range(batch_size):
            img = images[i]
            msk = masks[i]
            
            # Process individual image to (C, H, W) format
            proc_img = self._resize_image(img, size)
            proc_img = self._normalize_image(proc_img)
            
            # Process mask to (1, H, W) format
            proc_msk = self._resize_mask(msk, size)
            proc_msk = (proc_msk > 127.5).astype(np.float32)
            if len(proc_msk.shape) == 2:
                proc_msk = np.expand_dims(proc_msk, 0)  # Add channel dimension
                
            processed_images.append(proc_img)
            processed_masks.append(proc_msk)
        
        # Stack along batch dimension to get (B, C, H, W)
        batch_images = np.stack(processed_images, axis=0)
        batch_masks = np.stack(processed_masks, axis=0)
        
        # Verify shapes
        assert batch_images.shape[-2:] == size  # Height, Width should match target size
        assert batch_masks.shape[-2:] == size
        assert batch_images.shape[0] == batch_size
        assert batch_masks.shape[0] == batch_size
        
        return batch_images, batch_masks

    def _resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size"""
        if image.shape[:2] != size:
            # Preserve channel information
            channels = image.shape[2] if len(image.shape) == 3 else 1
            
            # OpenCV expects (width, height)
            resized = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
            
            # Ensure shape is (channels, height, width) for model input
            if len(resized.shape) == 2:
                resized = np.expand_dims(resized, 0)  # Add channel dim for grayscale
            else:
                # Move channels to first dimension
                resized = np.transpose(resized, (2, 0, 1))
                
            return resized
        
        # If no resize needed, still ensure channels-first format
        if len(image.shape) == 3:
            return np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, 0)

    def _resize_mask(self, mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize mask to target size"""
        if mask.shape[:2] != size:
            # OpenCV expects (width, height)
            mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image values"""
        # Ensure float32
        image = image.astype(np.float32) / 255.0
        
        # For HWC format (which is our input format)
        mean = np.array(self.config.mean, dtype=np.float32)
        std = np.array(self.config.std, dtype=np.float32)
        
        # Ensure image is in HWC format if not already
        if len(image.shape) == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Apply normalization
        mean = mean.reshape(1, 1, 3)
        std = std.reshape(1, 1, 3)
        normalized = (image - mean) / std
        
        # Convert to CHW format for output
        if len(normalized.shape) == 3:
            normalized = np.transpose(normalized, (2, 0, 1))
        
        return normalized

    def normalize_vgg_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor for VGG processing"""
        device = tensor.device
        mean = torch.tensor(self.config.mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(self.config.std, device=device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def postprocess(self, image: np.ndarray) -> Union[Image.Image, np.ndarray]:
        """Postprocess model output back to displayable image"""
        try:
            if image.ndim == 4:
                image = image[0]  # Remove batch dimension
            
            # Convert from CHW to HWC format
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            
            # Apply denormalization
            mean = np.array(self.config.mean)
            std = np.array(self.config.std)
            image = (image * std) + mean
            
            # Convert to uint8
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            
            return Image.fromarray(image)

        except Exception as e:
            logger.error(f"Error in postprocessing: {str(e)}")
            raise RuntimeError(f"Failed to postprocess output: {str(e)}")

    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor"""
        if isinstance(image, torch.Tensor):
            return image
            
        if image.ndim == 3:
            image = image.transpose(2, 0, 1)
        
        return torch.from_numpy(image).float()

    def from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array"""
        output = tensor.cpu().numpy()
        
        if output.ndim == 3:
            output = output.transpose(1, 2, 0)
            
        return output