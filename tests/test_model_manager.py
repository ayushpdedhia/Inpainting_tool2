# tests/test_model_manager.py

import pytest
import torch
import numpy as np
from PIL import Image
import os
from pathlib import Path
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from collections import OrderedDict
import yaml
from src.core.model_manager import ModelManager
from src.models.pconv.models.pconv_unet import PConvUNet
from src.models.pconv.loss import PConvLoss
from memory_profiler import memory_usage


class TestModelManager:
    """Test suite for ModelManager class"""

    @pytest.fixture
    def mock_weights_dir(self):
        """Fixture providing temporary weights directory with mock weights"""
        temp_dir = tempfile.mkdtemp()
        weights_dir = Path(temp_dir) / 'weights' / 'pconv'
        weights_dir.mkdir(parents=True)
        
        # Create mock weight directories
        (weights_dir / 'unet').mkdir()
        (weights_dir / 'vgg16').mkdir()
        
        # Create mock weight files
        mock_state_dict = OrderedDict([
            ('encoder.features.0.weight', torch.randn(64, 3, 7, 7)),
            ('encoder.features.0.bias', torch.randn(64)),
            ('decoder.conv1.weight', torch.randn(64, 64, 3, 3)),
            ('decoder.conv1.bias', torch.randn(64))
        ])
        
        torch.save(mock_state_dict, weights_dir / 'unet' / 'model_weights.pth')
        torch.save(mock_state_dict, weights_dir / 'vgg16' / 'vgg16_weights.pth')
        
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_config(self, mock_weights_dir):
        """Fixture providing temporary config file"""
        config_path = os.path.join(mock_weights_dir, 'config.yaml')
        
        config_content = {
            'model': {
                'name': 'pconv_unet',
                'weights_dir': 'weights/pconv',
                'input_size': [512, 512],
                'device': 'cpu'
            },
            'paths': {
                'data_dir': 'data',
                'weights_dir': 'weights',
                'temp_weights': 'temp_weights',
                'unet_weights': str(Path(mock_weights_dir) / 'weights' / 'pconv' / 'unet' / 'model_weights.pth'),
                'vgg_weights': str(Path(mock_weights_dir) / 'weights' / 'pconv' / 'vgg16' / 'vgg16_weights.pth')
            },
            'interface': {
                'canvas_size': 512,
                'max_image_size': 1024,
                'supported_formats': ['jpg', 'jpeg', 'png']
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f)
            
        return config_path

    @pytest.fixture
    def model_manager(self, mock_weights_dir):
        """Fixture providing ModelManager instance with mock weights"""
        with patch('src.core.model_manager.os.path.dirname') as mock_dirname:
            mock_dirname.return_value = str(Path(mock_weights_dir) / 'src' / 'core')
            return ModelManager()

    @pytest.fixture
    def test_image(self):
        """Fixture providing test image"""
        image = np.zeros((256, 256, 3), dtype=np.float32)
        # Add some patterns for testing
        image[50:150, 50:150] = 1.0  # White square
        image[100:200, 100:200, 0] = 1.0  # Red square
        return image

    @pytest.fixture
    def test_mask(self):
        """Fixture providing test mask"""
        mask = np.ones((256, 256), dtype=np.float32)
        mask[100:200, 100:200] = 0  # Area to inpaint
        return mask

    def test_initialization(self, model_manager):
        """Test ModelManager initialization"""
        assert isinstance(model_manager, ModelManager)
        assert hasattr(model_manager, 'models')
        assert hasattr(model_manager, 'device')
        assert 'partial convolutions' in model_manager.models
        assert isinstance(model_manager.models['partial convolutions'], PConvUNet)

    def test_model_loading(self, model_manager):
        """Test model loading functionality"""
        # Verify model is loaded correctly
        model = model_manager.models['partial convolutions']
        assert model.training is False  # Should be in eval mode
        assert next(model.parameters()).device == model_manager.device

    def test_preprocess_inputs(self, model_manager, test_image, test_mask):
        """Test input preprocessing"""
        image_tensor, mask_tensor = model_manager.preprocess_inputs(test_image, test_mask)
        
        # Check tensor properties
        assert isinstance(image_tensor, torch.Tensor)
        assert isinstance(mask_tensor, torch.Tensor)
        assert image_tensor.dim() == 4  # [B, C, H, W]
        assert mask_tensor.dim() == 4  # [B, 1, H, W]
        assert image_tensor.shape[1] == 3  # 3 channels
        assert mask_tensor.shape[1] == 1  # 1 channel

    def test_postprocess_output(self, model_manager):
        """Test output postprocessing"""
        # Create mock tensors
        output = torch.rand(1, 3, 256, 256)
        original = torch.rand(1, 3, 256, 256)
        mask = torch.ones(1, 1, 256, 256)
        
        result = model_manager.postprocess_output(output, original, mask)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (256, 256, 3)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_inpainting(self, model_manager, test_image, test_mask):
        """Test complete inpainting process"""
        try:
            result = model_manager.inpaint(test_image, test_mask)
            
            # Check result properties
            assert isinstance(result, np.ndarray)
            assert result.shape == test_image.shape
            assert result.dtype == np.float32
            assert np.all(result >= 0) and np.all(result <= 1)
            
            # Check if inpainting was performed
            masked_region = result[test_mask < 0.5]
            assert not np.all(masked_region == 0)
            
        except Exception as e:
            pytest.fail(f"Inpainting failed: {str(e)}")

    def test_available_models(self, model_manager):
        """Test available models listing"""
        models = model_manager.get_available_models()
        assert isinstance(models, dict)
        assert 'partial convolutions' in models
        assert models['partial convolutions'] == 'pdvgg16_bn'

    def test_model_info(self, model_manager):
        """Test model information retrieval"""
        info = model_manager.get_model_info('partial convolutions')
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'type' in info
        assert 'parameters' in info
        assert 'device' in info
        assert info['type'] == 'PConvUNet'

    def test_invalid_model_name(self, model_manager, test_image, test_mask):
        """Test handling of invalid model name"""
        with pytest.raises(ValueError):
            model_manager.inpaint(test_image, test_mask, model_name='nonexistent_model')

    def test_invalid_input_shapes(self, model_manager):
        """Test handling of invalid input shapes"""
        # Test invalid image shape
        invalid_image = np.zeros((100, 100))  # Missing channel dimension
        valid_mask = np.ones((100, 100))
        
        with pytest.raises(ValueError):
            model_manager.inpaint(invalid_image, valid_mask)
        
        # Test invalid mask shape
        valid_image = np.zeros((100, 100, 3))
        invalid_mask = np.ones((50, 50))  # Wrong size
        
        with pytest.raises(ValueError):
            model_manager.inpaint(valid_image, invalid_mask)

    def test_device_handling(self, model_manager, test_image, test_mask):
        """Test handling of different devices"""
        result = model_manager.inpaint(test_image, test_mask)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32  # Ensure result is of type float32



    def test_model_consistency(self, model_manager, test_image, test_mask):
        """Test model output consistency"""
        # Set random seeds for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)
        
        # Run inpainting twice with the same input
        result1 = model_manager.inpaint(test_image, test_mask)
        result2 = model_manager.inpaint(test_image, test_mask)
        
        # Results should be almost identical within a reasonable tolerance
        np.testing.assert_allclose(result1, result2, rtol=1e-5, atol=1e-8)


    def test_batch_processing(self, model_manager):
        """Test batch processing capabilities"""
        # Create batch of images and masks
        batch_size = 2
        images = np.stack([np.random.rand(256, 256, 3) for _ in range(batch_size)])
        masks = np.stack([np.ones((256, 256)) for _ in range(batch_size)])
        
        # Process each image individually
        results = []
        for i in range(batch_size):
            result = model_manager.inpaint(images[i], masks[i])
            results.append(result)
        
        assert len(results) == batch_size
        assert all(r.shape == (256, 256, 3) for r in results)

    def test_memory_efficiency(self, model_manager):
        """Test memory efficiency during inpainting"""
        def inpaint_large_image():
            large_image = np.random.rand(1024, 1024, 3).astype(np.float32)
            large_mask = np.ones((1024, 1024), dtype=np.float32)
            model_manager.inpaint(large_image, large_mask)
        
        mem_usage = memory_usage((inpaint_large_image,), interval=0.1)
        peak_memory = max(mem_usage) - min(mem_usage)
        
        # Assert that peak memory usage is below a certain threshold
        assert peak_memory < 2 * 1024  # Adjust based on observed usage

    def test_error_handling(self, model_manager):
        """Test error handling for various scenarios"""
        # Test with invalid image type
        with pytest.raises(TypeError):  # Changed from ValueError to TypeError
            model_manager.inpaint("not_an_image", np.zeros((100, 100)))
        
        # Test with incompatible image and mask sizes
        with pytest.raises(ValueError):  # This remains ValueError as it's a value/dimension error
            model_manager.inpaint(
                np.zeros((100, 100, 3)),
                np.zeros((200, 200))
            )
        
        # Test with invalid number of channels
        with pytest.raises(ValueError):  # This remains ValueError as it's a value error
            model_manager.inpaint(
                np.zeros((100, 100, 4)),  # 4 channels
                np.zeros((100, 100))
            )

    def test_output_validation(self, model_manager):
        """Test output validation for different input sizes"""
        test_sizes = [(128, 128), (256, 256), (512, 512)]
        
        for h, w in test_sizes:
            image = np.random.rand(h, w, 3).astype(np.float32)
            mask = np.ones((h, w), dtype=np.float32)
            mask[h//4:3*h//4, w//4:3*w//4] = 0
            
            result = model_manager.inpaint(image, mask)
            assert result.shape == (h, w, 3)
            assert np.all(result >= 0) and np.all(result <= 1)

if __name__ == '__main__':
    pytest.main([__file__])