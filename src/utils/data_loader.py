# src/utils/data_loader.py

import os
import cv2
from typing import Tuple, List, Optional, Union
import numpy as np
from PIL import Image
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path  # Add this import
from .mask_generator import MaskGenerator
from .image_processor import ImageProcessor, ProcessingConfig

logger = logging.getLogger(__name__)

class InpaintingDataset(Dataset):
    def __init__(self, 
                 image_dir: str,
                 mask_dir: Optional[str] = None,
                 image_size: Tuple[int, int] = (512, 512),
                 transform=None):
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory {image_dir} not found")
            
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform
        
        # Initialize processors
        self.image_processor = ImageProcessor(ProcessingConfig(target_size=image_size))
        self.mask_generator = MaskGenerator(height=image_size[0], width=image_size[1]) if mask_dir is None else None
        
        # Get image files
        self.image_files = self._get_files(image_dir)
        self.mask_files = self._get_files(mask_dir) if mask_dir else None
        
        logger.info(f"Found {len(self.image_files)} images in {image_dir}")
        if mask_dir:
            logger.info(f"Found {len(self.mask_files)} masks in {mask_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        if idx >= len(self.image_files):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.image_files)} images")

        try:
            # Load and verify image
            image_path = self.image_files[idx]
            if not os.path.exists(image_path):
                raise Exception(f"Image file not found: {image_path}")

            try:
                image = Image.open(image_path)
                image.verify()  # Verify image integrity
                image = Image.open(image_path).convert('RGB')  # Reopen for actual use
            except Exception as e:
                raise Exception(f"Invalid image file {image_path}: {str(e)}")

            # Handle mask
            if self.mask_dir:
                mask_idx = idx % len(self.mask_files)
                mask_path = self.mask_files[mask_idx]
                mask = np.array(Image.open(mask_path).convert('L'))
            else:
                mask = self.mask_generator.sample()
                mask = cv2.resize(mask, (image.size[0], image.size[1]), interpolation=cv2.INTER_NEAREST)

            # Process image and mask
            processed_image, processed_mask = self.image_processor.preprocess(image, mask)

            # Handle transform
            if self.transform is not None:
                if isinstance(processed_image, np.ndarray):
                    if len(processed_image.shape) == 4:
                        processed_image = processed_image[0]  # Remove batch dimension
                    # Convert to PIL Image for transforms
                    processed_image = Image.fromarray(
                        (processed_image.transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                # Set random seed based on index to ensure different transforms
                torch.manual_seed(idx)
                processed_image = self.transform(processed_image)

            return {
                'image': processed_image,
                'mask': processed_mask,
                'path': image_path
            }

        except Exception as e:
            logger.error(f"Error loading item at index {idx}: {str(e)}")
            raise

    @staticmethod
    def _get_files(directory: str) -> List[str]:
        """Get list of files in directory with image extensions"""
        if not directory or not os.path.exists(directory):
            return []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = []
        
        for f in sorted(os.listdir(directory)):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                files.append(os.path.join(directory, f))
                
        # Sort files to ensure consistent ordering
        files.sort()
        
        return files

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

__all__ = ['InpaintingDataset', 'get_data_loader']