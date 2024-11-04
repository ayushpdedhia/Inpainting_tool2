import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from src.core.model_manager import ModelManager
from src.utils.image_processor import ImageProcessor
from src.utils.mask_generator import MaskGenerator

class TestModelManager:
    @pytest.fixture
    def model_manager(self):
        """Initialize model manager for each test"""
        return ModelManager()
        
    @pytest.fixture
    def image_processor(self):
        """Initialize image processor for each test"""
        return ImageProcessor()
    
    @pytest.fixture
    def test_data_dir(self):
        """Get test data directory"""
        return Path("data/test_samples")
    
    def test_model_initialization(self, model_manager):
        """Test if model initializes correctly"""
        assert model_manager.models is not None
        assert 'partial convolutions' in model_manager.models
        assert isinstance(model_manager.models['partial convolutions'], torch.nn.Module)
    
    def test_model_device(self, model_manager):
        """Test if model is on correct device"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert model_manager.device == device
        
        # Check if model parameters are on correct device
        model = model_manager.models['partial convolutions']
        assert next(model.parameters()).device == device
    
    def test_available_models(self, model_manager):
        """Test available models listing"""
        models = model_manager.get_available_models()
        assert isinstance(models, dict)
        assert 'partial convolutions' in models
        
    @pytest.mark.parametrize("test_image", [
        "test_image_001.JPEG",  # Microscope image
        "test_image_002.JPEG",  # Purple flower
        "test_image_003.JPEG",  # Sea cucumber
        "test_image_004.JPEG",  # Magpie
        "test_image_005.JPEG"   # Vintage TV
    ])
    def test_inpainting_different_images(self, model_manager, image_processor, test_data_dir, test_image):
        """Test inpainting on different types of images"""
        # Load test image
        image_path = test_data_dir / "images" / test_image
        image = Image.open(image_path)
        image_np = np.array(image) / 255.0  # Normalize to [0, 1]
        
        # Create a simple test mask
        mask = np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
        mask[100:200, 100:200] = 0  # Create a hole
        
        # Run inpainting
        try:
            result = model_manager.inpaint(image_np, mask)
            
            # Verify result
            assert isinstance(result, np.ndarray)
            assert result.shape == image_np.shape
            assert not np.any(np.isnan(result))
            assert result.min() >= 0 and result.max() <= 1
            
            # Verify hole region has been filled
            hole_region = result[100:200, 100:200]
            assert not np.all(hole_region == 0)
            
        except Exception as e:
            pytest.fail(f"Inpainting failed for {test_image}: {str(e)}")
    
    @pytest.mark.parametrize("mask_file", [
        "mask_large.png",
        "mask_edge.png",
        "mask_thick.png",
        "mask_thin.png",
        "mask_corner.png",
        "mask_small.png"
    ])
    def test_inpainting_different_masks(self, model_manager, image_processor, test_data_dir, mask_file):
        """Test inpainting with different mask types"""
        # Load test image (using first test image for all masks)
        image_path = test_data_dir / "images" / "test_image_001.JPEG"
        image = Image.open(image_path)
        image_np = np.array(image) / 255.0
        
        # Load mask
        mask_path = test_data_dir / "masks" / mask_file
        mask = np.array(Image.open(mask_path).convert('L')) / 255.0
        
        # Resize mask if needed
        if mask.shape[:2] != image_np.shape[:2]:
            mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
                image_np.shape[:2][::-1], Image.NEAREST)) / 255.0
        
        try:
            result = model_manager.inpaint(image_np, mask)
            
            # Verify result
            assert isinstance(result, np.ndarray)
            assert result.shape == image_np.shape
            assert not np.any(np.isnan(result))
            assert result.min() >= 0 and result.max() <= 1
            
            # Verify holes have been filled
            hole_regions = result[mask < 0.5]
            assert not np.all(hole_regions == 0)
            
        except Exception as e:
            pytest.fail(f"Inpainting failed for {mask_file}: {str(e)}")
    
    def test_model_consistency(self, model_manager, image_processor, test_data_dir):
        """Test if model produces consistent results for same input"""
        # Load test image
        image_path = test_data_dir / "images" / "test_image_001.JPEG"
        image = Image.open(image_path)
        image_np = np.array(image) / 255.0
        
        # Create simple mask
        mask = np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
        mask[100:200, 100:200] = 0
        
        # Run inpainting twice
        result1 = model_manager.inpaint(image_np, mask)
        result2 = model_manager.inpaint(image_np, mask)
        
        # Check results are identical
        np.testing.assert_array_almost_equal(result1, result2)

    def test_error_handling(self, model_manager):
        """Test error handling for invalid inputs"""
        # Test invalid image
        with pytest.raises(ValueError):
            model_manager.inpaint(np.zeros((100, 100)), np.zeros((50, 50)))
        
        # Test invalid mask
        with pytest.raises(ValueError):
            model_manager.inpaint(np.zeros((100, 100, 3)), np.zeros((50, 50)))
        
        # Test invalid model name
        with pytest.raises(ValueError):
            model_manager.inpaint(np.zeros((100, 100, 3)), np.zeros((100, 100)), 
                                model_name="nonexistent_model")