# src/utils/mask_generator.py

import os
import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import random
import logging

logger = logging.getLogger(__name__)

@dataclass
class MaskConfig:
    """Configuration for mask generation parameters"""
    min_num_shapes: int = 1
    max_num_shapes: int = 20
    min_shape_size: int = 3
    max_shape_size: int = None  # Will be set based on image dimensions
    shape_probability: dict = None  # Probability distribution for different shapes

    def __post_init__(self):
        if self.shape_probability is None:
            self.shape_probability = {
                'line': 0.4,
                'circle': 0.3,
                'ellipse': 0.3
            }

class MaskGenerator:
    """Generates random irregular masks for image inpainting"""

    def __init__(self, 
                 height: int, 
                 width: int, 
                 channels: int = 1,
                 config: Optional[MaskConfig] = None,
                 rand_seed: Optional[int] = None,
                 mask_dir: Optional[str] = None):
        """
        Initialize the mask generator.
        
        Args:
            height: Height of the masks to generate
            width: Width of the masks to generate
            channels: Number of channels in the masks
            config: Configuration for mask generation
            rand_seed: Optional random seed for reproducibility
            mask_dir: Optional directory with predefined masks
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.config = config or MaskConfig()
        self.mask_dir = mask_dir
        
        # Set size scale based on image dimensions
        self.config.max_shape_size = self.config.max_shape_size or int((width + height) * 0.03)
        
        # Set random seed if provided
        if rand_seed is not None:
            random.seed(rand_seed)
            np.random.seed(rand_seed)
        
        # Load mask files if directory provided
        self.mask_files = self._load_mask_files() if mask_dir else []
        if mask_dir:
            logger.info(f"Found {len(self.mask_files)} masks in {mask_dir}")

    def sample(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate or load a random mask.
        
        Args:
            random_seed: Optional random seed for this specific sample
            
        Returns:
            A binary mask array of shape (height, width, channels)
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        if self.mask_files:
            return self._load_random_mask()
        else:
            return self._generate_random_mask()

    def _generate_random_mask(self) -> np.ndarray:
        """Generate a random irregular mask using OpenCV"""
        mask = np.zeros((self.height, self.width, self.channels), np.uint8)
        
        # Generate random shapes
        num_shapes = random.randint(self.config.min_num_shapes, self.config.max_num_shapes)
        
        for _ in range(num_shapes):
            # Randomly select shape type based on probability distribution
            shape_type = random.choices(
                list(self.config.shape_probability.keys()),
                list(self.config.shape_probability.values())
            )[0]
            
            # Generate shape
            if shape_type == 'line':
                self._draw_random_line(mask)
            elif shape_type == 'circle':
                self._draw_random_circle(mask)
            elif shape_type == 'ellipse':
                self._draw_random_ellipse(mask)
        
        return 1 - mask

    def _draw_random_line(self, mask: np.ndarray):
        """Draw a random line on the mask"""
        x1, x2 = random.randint(1, self.width), random.randint(1, self.width)
        y1, y2 = random.randint(1, self.height), random.randint(1, self.height)
        thickness = random.randint(self.config.min_shape_size, self.config.max_shape_size)
        cv2.line(mask, (x1, y1), (x2, y2), (1,) * self.channels, thickness)

    def _draw_random_circle(self, mask: np.ndarray):
        """Draw a random circle on the mask"""
        x1, y1 = random.randint(1, self.width), random.randint(1, self.height)
        radius = random.randint(self.config.min_shape_size, self.config.max_shape_size)
        cv2.circle(mask, (x1, y1), radius, (1,) * self.channels, -1)

    def _draw_random_ellipse(self, mask: np.ndarray):
        """Draw a random ellipse on the mask"""
        x1, y1 = random.randint(1, self.width), random.randint(1, self.height)
        s1, s2 = random.randint(1, self.width), random.randint(1, self.height)
        a1, a2, a3 = [random.randint(3, 180) for _ in range(3)]
        thickness = random.randint(self.config.min_shape_size, self.config.max_shape_size)
        cv2.ellipse(mask, (x1, y1), (s1, s2), a1, a2, a3, (1,) * self.channels, thickness)

    def _load_mask_files(self) -> List[str]:
        """Load all mask files from the specified directory"""
        if not self.mask_dir or not os.path.exists(self.mask_dir):
            return []
            
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        return [
            os.path.join(self.mask_dir, f) 
            for f in os.listdir(self.mask_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]

    def _load_random_mask(self) -> np.ndarray:
        """Load and process a random mask from the mask directory"""
        mask_path = random.choice(self.mask_files)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize if necessary
        if mask.shape[:2] != (self.height, self.width):
            mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # Binarize
        mask = (mask > 127.5).astype(np.uint8)
        
        # Add channels dimension if needed
        if self.channels > 1:
            mask = np.stack([mask] * self.channels, axis=-1)
            
        return mask