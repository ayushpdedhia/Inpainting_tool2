# D:\Inpainting_tool2\tests\test_mask_generator.py
# cd D:\Inpainting_tool2
# python -m pytest tests/test_mask_generator.py
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

from src.utils.mask_generator import MaskGenerator, MaskConfig

class TestMaskGenerator:
    @pytest.fixture
    def mask_generator(self):
        """Initialize mask generator with default settings"""
        return MaskGenerator(height=512, width=512)
    
    @pytest.fixture
    def test_data_dir(self):
        """Get test data directory"""
        return Path("data/test_samples")
    
    def test_initialization(self):
        """Test mask generator initialization with different parameters"""
        # Test default initialization
        mg = MaskGenerator(512, 512)
        assert mg.height == 512
        assert mg.width == 512
        assert mg.channels == 1
        
        # Test custom config
        custom_config = MaskConfig(
            min_num_shapes=5,
            max_num_shapes=10,
            min_shape_size=10,
            max_shape_size=50
        )
        mg = MaskGenerator(512, 512, config=custom_config)
        assert mg.config.min_num_shapes == 5
        assert mg.config.max_shape_size == 50
    
    def test_sample_generation(self, mask_generator):
        """Test random mask sampling"""
        # Generate sample mask
        mask = mask_generator.sample()
        
        # Verify mask properties
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.uint8
        assert np.all((mask == 0) | (mask == 1))  # Binary mask
        assert np.any(mask == 0)  # Has holes
        assert np.any(mask == 1)  # Has valid regions
    
    def test_reproducibility(self):
        """Test mask generation reproducibility with seed"""
        mg1 = MaskGenerator(512, 512, rand_seed=42)
        mg2 = MaskGenerator(512, 512, rand_seed=42)
        
        mask1 = mg1.sample()
        mask2 = mg2.sample()
        
        np.testing.assert_array_equal(mask1, mask2)
    
    @pytest.mark.parametrize("mask_file", [
        "mask_large.png",
        "mask_edge.png",
        "mask_thick.png",
        "mask_thin.png",
        "mask_corner.png",
        "mask_small.png"
    ])
    def test_mask_properties(self, test_data_dir, mask_file):
        """Test properties of provided test masks"""
        mask_path = test_data_dir / "masks" / mask_file
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # Verify basic properties
        assert isinstance(mask, np.ndarray)
        assert len(mask.shape) == 2  # 2D mask
        assert mask.dtype == np.uint8
        assert np.all((mask >= 0) & (mask <= 255))
        
        # Convert to binary and check hole properties
        binary_mask = (mask > 127).astype(np.uint8)
        assert np.any(binary_mask == 0)  # Has holes
        assert np.any(binary_mask == 1)  # Has valid regions
    
    def test_shape_generation(self, mask_generator):
        """Test individual shape drawing functions"""
        mask = np.ones((512, 512), dtype=np.uint8)
        
        # Test line drawing
        mask_generator._draw_random_line(mask)
        assert np.any(mask == 1)
        
        # Test circle drawing
        mask = np.ones((512, 512), dtype=np.uint8)
        mask_generator._draw_random_circle(mask)
        assert np.any(mask == 1)
        
        # Test ellipse drawing
        mask = np.ones((512, 512), dtype=np.uint8)
        mask_generator._draw_random_ellipse(mask)
        assert np.any(mask == 1)
    
    def test_mask_variety(self, mask_generator):
        """Test if generated masks are sufficiently different"""
        masks = [mask_generator.sample() for _ in range(5)]
        
        # Compare each pair of masks
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                difference = np.sum(masks[i] != masks[j])
                # Ensure masks are not too similar
                assert difference > 1000, f"Masks {i} and {j} are too similar"
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid shape probabilities
        invalid_config = MaskConfig(
            shape_probability={'line': 0.5, 'circle': 0.2, 'ellipse': 0.3}  # Sum = 1.0
        )
        invalid_config.shape_probability = {'line': 0.5, 'circle': 0.2}  # Now sum < 1.0
        
        with pytest.raises(ValueError, match="Shape probabilities must sum to 1.0"):
            MaskGenerator(512, 512, config=invalid_config)
        
        # Test invalid size parameters
        invalid_config = MaskConfig(
            min_shape_size=100,
            max_shape_size=50  # max < min
        )
        with pytest.raises(ValueError, match="min_shape_size cannot be greater than max_shape_size"):
            MaskGenerator(512, 512, config=invalid_config)
        
        # Test invalid shape numbers
        invalid_config = MaskConfig(
            min_num_shapes=20,
            max_num_shapes=10  # max < min
        )
        with pytest.raises(ValueError, match="min_num_shapes cannot be greater than max_num_shapes"):
            MaskGenerator(512, 512, config=invalid_config)
    
    def test_mask_loading(self, test_data_dir):
        """Test loading and processing predefined masks"""
        mg = MaskGenerator(512, 512, mask_dir=str(test_data_dir / "masks"))
        
        # Test random mask loading
        loaded_mask = mg._load_random_mask()
        assert isinstance(loaded_mask, np.ndarray)
        assert loaded_mask.shape == (512, 512)
        assert loaded_mask.dtype == np.uint8
        
        # Test mask processing
        assert np.all((loaded_mask == 0) | (loaded_mask == 1))  # Binary
        
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test invalid dimensions
        with pytest.raises(ValueError):
            MaskGenerator(-100, 100)
        
        # Test invalid channel number
        with pytest.raises(ValueError):
            MaskGenerator(512, 512, channels=0)
        
        # Test invalid mask directory
        with pytest.raises(ValueError):
            MaskGenerator(512, 512, mask_dir="nonexistent_directory")