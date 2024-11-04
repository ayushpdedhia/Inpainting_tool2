# tests/test_image_processor.py

import pytest
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import cv2
from dataclasses import asdict
from src.utils.image_processor import ImageProcessor, ProcessingConfig

class TestImageProcessor:
    """Test suite for image processing functionality"""

    @pytest.fixture
    def test_image(self) -> np.ndarray:
        """Fixture providing a test image"""
        # Create a simple RGB test image (100x100)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some patterns for better testing
        image[20:40, 20:40] = [255, 0, 0]  # Red square
        image[60:80, 60:80] = [0, 255, 0]  # Green square
        return image

    @pytest.fixture
    def test_mask(self) -> np.ndarray:
        """Fixture providing a test mask"""
        # Create a simple binary mask (100x100)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255  # White square in center
        return mask

    @pytest.fixture
    def processor(self) -> ImageProcessor:
        """Fixture providing default ImageProcessor instance"""
        return ImageProcessor()

    @pytest.fixture
    def custom_config(self) -> ProcessingConfig:
        """Fixture providing custom processing configuration"""
        return ProcessingConfig(
            target_size=(256, 256),
            normalize_range=(0.0, 1.0),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )

    def test_processing_config_initialization(self):
        """Test ProcessingConfig initialization and defaults"""
        config = ProcessingConfig()
        
        assert config.target_size == (512, 512)
        assert config.normalize_range == (0.0, 1.0)
        assert config.mean == (0.485, 0.456, 0.406)
        assert config.std == (0.229, 0.224, 0.225)
        
        # Test custom values
        custom_config = ProcessingConfig(
            target_size=(256, 256),
            normalize_range=(-1.0, 1.0),
            mean=(0.5, 0.5, 0.5),
            std=(0.2, 0.2, 0.2)
        )
        
        assert custom_config.target_size == (256, 256)
        assert custom_config.normalize_range == (-1.0, 1.0)
        assert custom_config.mean == (0.5, 0.5, 0.5)
        assert custom_config.std == (0.2, 0.2, 0.2)

    def test_processor_initialization(self):
        """Test ImageProcessor initialization"""
        # Test default initialization
        processor = ImageProcessor()
        assert isinstance(processor.config, ProcessingConfig)
        
        # Test custom config initialization
        custom_config = ProcessingConfig(target_size=(256, 256))
        processor = ImageProcessor(custom_config)
        assert processor.config.target_size == (256, 256)

    def test_preprocess_pil_image(self, processor, test_image):
        """Test preprocessing with PIL Image input"""
        pil_image = Image.fromarray(test_image)
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        processed_image, processed_mask = processor.preprocess(pil_image, mask)
        
        assert isinstance(processed_image, np.ndarray)
        assert processed_image.shape[0] == 1  # Batch dimension
        assert processed_image.shape[-2:] == processor.config.target_size

    def test_preprocess_numpy_array(self, processor, test_image, test_mask):
        """Test preprocessing with numpy array input"""
        processed_image, processed_mask = processor.preprocess(test_image, test_mask)
        
        assert isinstance(processed_image, np.ndarray)
        assert isinstance(processed_mask, np.ndarray)
        assert processed_image.shape[0] == 1  # Batch dimension
        assert processed_mask.shape[0] == 1  # Batch dimension
        assert processed_image.dtype == np.float32
        assert processed_mask.dtype == np.float32

    def test_custom_target_size(self, processor, test_image, test_mask):
        """Test preprocessing with custom target size"""
        custom_size = (256, 256)
        processed_image, processed_mask = processor.preprocess(
            test_image, test_mask, target_size=custom_size
        )
        
        assert processed_image.shape[-2:] == custom_size
        assert processed_mask.shape[-2:] == custom_size

    def test_image_normalization(self, processor, test_image):
        processed_image, _ = processor.preprocess(test_image, np.zeros_like(test_image[:,:,0]))
        
        # Check value range
        assert np.all(processed_image >= -5) and np.all(processed_image <= 5)
        
        # Test if normalization with ImageNet stats was applied
        original_mean = np.mean(test_image, axis=(0, 1)) / 255.0  # Shape: (3,)
        
        # Get the processed mean, ensuring we're comparing the same shapes
        processed = processed_image[0]  # Remove batch dimension
        processed_mean = np.mean(processed, axis=(1, 2))  # Take mean across H,W dimensions
        
        # Now both means should be shape (3,)
        assert processed_mean.shape == original_mean.shape
        assert not np.allclose(original_mean, processed_mean)

    def test_mask_binarization(self, processor, test_mask):
        """Test mask binarization"""
        _, processed_mask = processor.preprocess(
            np.zeros((100, 100, 3), dtype=np.uint8), 
            test_mask
        )
        
        # Check if mask is binary
        unique_values = np.unique(processed_mask)
        assert len(unique_values) <= 2
        assert np.all((unique_values >= 0) & (unique_values <= 1))

    def test_vgg_normalization(self, processor):
        """Test VGG tensor normalization"""
        test_tensor = torch.randn(1, 3, 64, 64)
        normalized = processor.normalize_vgg_tensor(test_tensor)
        
        assert normalized.shape == test_tensor.shape
        assert normalized.device == test_tensor.device
        
        # Check if normalization was applied correctly
        expected_mean = torch.tensor(processor.config.mean).view(1, 3, 1, 1)
        expected_std = torch.tensor(processor.config.std).view(1, 3, 1, 1)
        manually_normalized = (test_tensor - expected_mean) / expected_std
        assert torch.allclose(normalized, manually_normalized)

    def test_postprocess(self, processor, test_image):
        """Test postprocessing"""
        # First preprocess the image
        processed_image, _ = processor.preprocess(test_image, np.zeros_like(test_image[:,:,0]))
        
        # Then postprocess
        result = processor.postprocess(processed_image)
        
        assert isinstance(result, Image.Image)
        assert result.size == processor.config.target_size[::-1]  # PIL uses (width, height)
        
        # Check value range
        result_array = np.array(result)
        assert result_array.dtype == np.uint8
        assert np.all(result_array >= 0) and np.all(result_array <= 255)

    def test_tensor_conversion(self, processor, test_image):
        """Test tensor conversion methods"""
        # Test to_tensor
        tensor = processor.to_tensor(test_image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 3  # CHW format
        assert tensor.dtype == torch.float32
        
        # Test from_tensor
        array = processor.from_tensor(tensor)
        assert isinstance(array, np.ndarray)
        assert array.shape == test_image.shape  # Back to HWC format

    def test_error_handling(self, processor):
        """Test error handling for invalid inputs"""
        # Test invalid image shape
        invalid_image = np.zeros((100, 100, 4))  # 4 channels
        invalid_mask = np.zeros((50, 50))  # Wrong size
        
        with pytest.raises(RuntimeError):
            processor.preprocess(invalid_image, invalid_mask)
        
        # Test invalid mask shape
        with pytest.raises(RuntimeError):
            processor.preprocess(np.zeros((100, 100, 3)), np.zeros((50, 50, 2)))

    def test_resize_consistency(self, processor, test_image, test_mask):
        """Test consistency of resizing operations"""
        # Process same input twice
        result1_img, result1_mask = processor.preprocess(test_image, test_mask)
        result2_img, result2_mask = processor.preprocess(test_image, test_mask)
        
        assert np.array_equal(result1_img, result2_img)
        assert np.array_equal(result1_mask, result2_mask)

    def test_batch_handling(self, processor, test_image, test_mask):
        """Test handling of batched inputs"""
        # Create batch of images and masks
        batch_size = 3
        batched_images = np.stack([test_image] * batch_size)
        batched_masks = np.stack([test_mask] * batch_size)
        
        processed_images, processed_masks = processor.preprocess(
            batched_images, batched_masks
        )
        
        assert processed_images.shape[0] == batch_size
        assert processed_masks.shape[0] == batch_size

    def test_edge_cases(self, processor):
        """Test edge cases and boundary conditions"""
        # Test tiny image
        tiny_image = np.zeros((1, 1, 3), dtype=np.uint8)
        tiny_mask = np.zeros((1, 1), dtype=np.uint8)
        processed_image, processed_mask = processor.preprocess(tiny_image, tiny_mask)
        assert processed_image.shape[-2:] == processor.config.target_size
        
        # Test large image
        large_image = np.zeros((2000, 2000, 3), dtype=np.uint8)
        large_mask = np.zeros((2000, 2000), dtype=np.uint8)
        processed_image, processed_mask = processor.preprocess(large_image, large_mask)
        assert processed_image.shape[-2:] == processor.config.target_size

    def test_memory_efficiency(self, processor, test_image, test_mask):
        """Test memory efficiency"""
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process large batch of images
        batch_size = 10
        large_batch = np.stack([test_image] * batch_size)
        large_masks = np.stack([test_mask] * batch_size)
        
        _ = processor.preprocess(large_batch, large_masks)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 1GB for this test)
        assert memory_increase < 1024 * 1024 * 1024

if __name__ == '__main__':
    pytest.main([__file__])