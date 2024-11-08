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
    """
    Generates random irregular masks for image inpainting.
    
    Mask Convention:
    - Generated mask values:
        0 = areas to inpaint (holes)
        1 = areas to keep (valid regions)
    - When loading external masks:
        white (255) = areas to inpaint
        black (0) = areas to keep
    """

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
        # Input validation
        if height <= 0 or width <= 0:
            raise ValueError("Height and width must be positive")
        if channels <= 0:
            raise ValueError("Number of channels must be positive")
        if mask_dir and not os.path.exists(mask_dir):
            raise ValueError(f"Mask directory does not exist: {mask_dir}")
        
        self.height = height
        self.width = width
        self.channels = channels
        self.config = config or MaskConfig()
        self.mask_dir = mask_dir
        
        # Set size scale based on image dimensions
        self.config.max_shape_size = self.config.max_shape_size or int((width + height) * 0.03)

        # Validate configuration
        self._validate_config()
        
        # Initialize random number generator
        self.rng = np.random.RandomState(rand_seed)
        if rand_seed is not None:
            random.seed(rand_seed)
            np.random.seed(rand_seed)
        
        # Load mask files if directory provided
        self.mask_files = self._load_mask_files() if mask_dir else []
        if mask_dir:
            logger.info(f"Found {len(self.mask_files)} masks in {mask_dir}")

    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate shape parameters
        if self.config.min_shape_size > self.config.max_shape_size:
            raise ValueError("min_shape_size cannot be greater than max_shape_size")
        if self.config.min_num_shapes > self.config.max_num_shapes:
            raise ValueError("min_num_shapes cannot be greater than max_num_shapes")
        if self.config.min_shape_size < 1:
            raise ValueError("min_shape_size must be positive")
            
        # Validate probabilities
        total_prob = sum(self.config.shape_probability.values())
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError("Shape probabilities must sum to 1.0")

    def sample(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate or load a random mask.
        
        Args:
            random_seed: Optional random seed for this specific sample
            
        Returns:
            np.ndarray: Binary mask where:
                       0 = areas to inpaint
                       1 = areas to keep original
        """
        if random_seed is not None:
            self.rng = np.random.RandomState(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

        if self.mask_files:
            mask = self._load_random_mask()
        else:
            mask = self._generate_random_mask()

        # Debug output
        print("\n=== Generated Mask Stats ===")
        print(f"Shape: {mask.shape}")
        print(f"Range: [{mask.min()}, {mask.max()}]")
        print(f"Unique values: {np.unique(mask)}")
        print(f"Inpaint area: {np.mean(mask == 0) * 100:.1f}%")
        
        return mask.squeeze()
        

    def _generate_random_mask(self) -> np.ndarray:
        """
        Generate a random irregular mask.
        
        Returns:
            np.ndarray: Binary mask where:
                       0 = inpaint regions (holes)
                       1 = keep original
        """
        # Initialize mask with all ones (keep everything)
        mask = np.ones((self.height, self.width, self.channels), np.uint8)
        
        # Generate random shapes
        num_shapes = self.rng.randint(self.config.min_num_shapes, self.config.max_num_shapes + 1)
        print(f"\nGenerating {num_shapes} random shapes")
        
        for i in range(num_shapes):
            shape_type = self.rng.choice(
                list(self.config.shape_probability.keys()),
                p=list(self.config.shape_probability.values())
            )
            
            # Draw shape (sets pixels to 0 where we want to inpaint)
            if shape_type == 'line':
                self._draw_random_line(mask)
            elif shape_type == 'circle':
                self._draw_random_circle(mask)
            elif shape_type == 'ellipse':
                self._draw_random_ellipse(mask)

            # Debug output for each shape
            if i == 0 or i == num_shapes-1:  # Only print first and last for brevity
                print(f"Shape {i+1}: {shape_type}")
                print(f"Mask range: [{mask.min()}, {mask.max()}]")
                print(f"Inpaint area: {np.mean(mask == 0) * 100:.1f}%")

        
        return mask

    def _draw_random_line(self, mask: np.ndarray):
        """Draw a random line on the mask (sets line pixels to 0)"""
        x1 = self.rng.randint(1, self.width)
        x2 = self.rng.randint(1, self.width)
        y1 = self.rng.randint(1, self.height)
        y2 = self.rng.randint(1, self.height)
        thickness = self.rng.randint(self.config.min_shape_size, self.config.max_shape_size + 1)
        cv2.line(mask, (x1, y1), (x2, y2), (0,) * self.channels, thickness)  # Use 0 for inpaint

    def _draw_random_circle(self, mask: np.ndarray):
        """Draw a random circle on the mask (sets circle pixels to 0)"""
        x1 = self.rng.randint(1, self.width)
        y1 = self.rng.randint(1, self.height)
        radius = self.rng.randint(self.config.min_shape_size, self.config.max_shape_size + 1)
        cv2.circle(mask, (x1, y1), radius, (0,) * self.channels, -1)  # Use 0 for inpaint

    def _draw_random_ellipse(self, mask: np.ndarray):
        """Draw a random ellipse on the mask (sets ellipse pixels to 0)"""
        x1 = self.rng.randint(1, self.width)
        y1 = self.rng.randint(1, self.height)
        s1 = self.rng.randint(1, self.width)
        s2 = self.rng.randint(1, self.height)
        a1, a2, a3 = [self.rng.randint(3, 180) for _ in range(3)]
        thickness = self.rng.randint(self.config.min_shape_size, self.config.max_shape_size + 1)
        cv2.ellipse(mask, (x1, y1), (s1, s2), a1, a2, a3, (0,) * self.channels, thickness)  # Use 0 for inpaint

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
        """
        Load and process a random mask from the mask directory.
        
        Returns:
            np.ndarray: Binary mask where:
                       0 = inpaint regions (holes)
                       1 = keep original
        """
        mask_path = random.choice(self.mask_files)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"\n=== Loading Mask: {os.path.basename(mask_path)} ===")
        print(f"Original shape: {mask.shape}")
        print(f"Original range: [{mask.min()}, {mask.max()}]")
        
        # Resize if necessary
        if mask.shape[:2] != (self.height, self.width):
            mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # Convert white (255) to 0 (inpaint) and black (0) to 1 (keep)
        mask = (mask < 127.5).astype(np.uint8)
        
        # Add channels if needed
        if self.channels > 1:
            mask = np.stack([mask] * self.channels, axis=-1)
        
        print(f"Processed shape: {mask.shape}")
        print(f"Processed range: [{mask.min()}, {mask.max()}]")
        print(f"Inpaint area: {np.mean(mask == 0) * 100:.1f}%")
        
        return mask