# src/utils/data_loader.py

import os
import cv2
from typing import Tuple, List, Optional, Union
import numpy as np
from PIL import Image
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from .mask_generator import MaskGenerator
from .image_processor import ImageProcessor, ProcessingConfig

logger = logging.getLogger(__name__)

class InpaintingDataset(Dataset):
    """Dataset class for image inpainting training"""
    
    def __init__(self, 
                 image_dir: str,
                 mask_dir: Optional[str] = None,
                 image_size: Tuple[int, int] = (512, 512),
                 transform=None):
        """
        Args:
            image_dir: Directory containing training images
            mask_dir: Optional directory containing masks, if None generates random masks
            image_size: Target size for images
            transform: Optional transform to be applied on a sample
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform
        
        # Initialize processors
        self.image_processor = ImageProcessor(ProcessingConfig(target_size=image_size))
        self.mask_generator = MaskGenerator(*image_size) if mask_dir is None else None
        
        # Get image files
        self.image_files = self._get_files(image_dir)
        self.mask_files = self._get_files(mask_dir) if mask_dir else None
        
        logger.info(f"Found {len(self.image_files)} images in {image_dir}")
        if mask_dir:
            logger.info(f"Found {len(self.mask_files)} masks in {mask_dir}")

    def __len__(self) -> int:
        return len(self.image_files)



    def __getitem__(self, idx: int) -> dict:
        # Load image
        try:
            image_path = self.image_files[idx]
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise Exception(f"Failed to load image {image_path}: {str(e)}")
        
        # Get mask
        if self.mask_dir:
            mask_path = self.mask_files[idx % len(self.mask_files)]
            mask = np.array(Image.open(mask_path).convert('L'))
        else:
            # Generate mask with same dimensions as image
            mask = self.mask_generator.sample()
            # Ensure mask matches image dimensions
            mask = cv2.resize(mask, (image.size[0], image.size[1]), interpolation=cv2.INTER_NEAREST)
        
        # Process image and mask
        processed_image, processed_mask = self.image_processor.preprocess(image, mask)
                
        if self.transform:
            # Ensure image is in correct format (B,C,H,W) -> (C,H,W)
            if len(processed_image.shape) == 4:
                processed_image = processed_image[0]  # Remove batch dimension
            processed_image = Image.fromarray((processed_image.transpose(1, 2, 0) * 255).astype(np.uint8))
            processed_image = self.transform(processed_image)
        
        return {
            'image': processed_image,
            'mask': processed_mask,
            'path': image_path
        }
        
    @staticmethod
    def _get_files(directory: str) -> List[str]:
        """Get list of files in directory with image extensions"""
        if not directory:
            return []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = []
        for f in os.listdir(directory):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                files.append(os.path.join(directory, f))
        
        # Remove duplicate extensions (e.g., both .jpg and .jpeg)
        return list(set(files))


def get_data_loader(image_dir: str,
                mask_dir: Optional[str] = None,
                batch_size: int = 8,
                image_size: Tuple[int, int] = (512, 512),
                num_workers: int = 4,
                shuffle: bool = True) -> DataLoader:
    """Create data loader for training/validation"""
    
    dataset = InpaintingDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=image_size
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Add shuffle attribute to loader
    setattr(loader, 'shuffle', shuffle)
    
    return loader