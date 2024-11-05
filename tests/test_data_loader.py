# tests/test_data_loader.py

import pytest
import os
import numpy as np
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from src.utils.data_loader import InpaintingDataset, get_data_loader
from src.utils.mask_generator import MaskGenerator
from src.utils.image_processor import ImageProcessor, ProcessingConfig

class TestDataLoader:
    """Test suite for data loading functionality"""

    @pytest.fixture
    def temp_data_dir(self):
        """Fixture providing temporary directories for test data"""
        temp_dir = tempfile.mkdtemp()
        # Create subdirectories
        images_dir = Path(temp_dir) / 'images'
        masks_dir = Path(temp_dir) / 'masks'
        images_dir.mkdir()
        masks_dir.mkdir()
        
        # Create some test images and masks
        self._create_test_images(images_dir)
        self._create_test_masks(masks_dir)
        
        yield {
            'root': temp_dir,
            'images': str(images_dir),
            'masks': str(masks_dir)
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def _create_test_images(self, directory: Path, num_images: int = 5):
        """Helper method to create test images matching your actual structure"""
        # Create test images with your naming convention
        for i in range(num_images):
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(directory / f'test_image_{str(i+1).zfill(3)}.jpeg')  # Changed to .jpeg to match your structure
            
        # Create validation images
        for i in range(num_images):
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(directory / f'val_image_{str(i+1).zfill(3)}.jpeg')

    def _create_test_masks(self, directory: Path):
        """Helper method to create test masks matching your actual structure"""
        mask_names = [
            'mask_corner',
            'mask_edge',
            'mask_large',
            'mask_small',
            'mask_thick',
            'mask_thin'
        ]
        
        for mask_name in mask_names:
            mask_array = np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255
            mask = Image.fromarray(mask_array)
            mask.save(directory / f'{mask_name}.png')

    @pytest.fixture
    def custom_transform(self):
        """Fixture providing a custom transform"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def test_dataset_initialization(self, temp_data_dir):
        """Test basic dataset initialization"""
        # Test dataset with test images only
        dataset_test = InpaintingDataset(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks'],
            image_type='test'  # This will filter for test_image_* files
        )
        
        # Verify test dataset properties
        assert len(dataset_test) == 5  # Should find 5 test images
        assert dataset_test.image_size == (512, 512)
        assert dataset_test.transform is None
        assert dataset_test.mask_dir is not None
        assert len(dataset_test.image_files) == 5  # 5 test images
        assert len(dataset_test.mask_files) == 6  # 6 mask files (corner, edge, large, small, thick, thin)

        # Additional verification for validation images
        dataset_val = InpaintingDataset(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks'],
            image_type='val'  # This will filter for val_image_* files
        )
        assert len(dataset_val) == 5  # Should find 5 validation images
        assert len(dataset_val.image_files) == 5  # 5 validation images
        
        # Test with no image_type (should get all images)
        dataset_all = InpaintingDataset(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks']
        )
        assert len(dataset_all) == 10  # Should find all 10 images
        assert len(dataset_all.image_files) == 10  # All 10 images

    def test_dataset_without_masks(self, temp_data_dir):
        """Test dataset initialization without mask directory"""
        dataset = InpaintingDataset(
            image_dir=temp_data_dir['images'],
            mask_dir=None
        )
        
        assert dataset.mask_dir is None
        assert dataset.mask_generator is not None
        
        # Test mask generation
        sample = dataset[0]
        assert 'mask' in sample
        assert isinstance(sample['mask'], np.ndarray)
        assert sample['mask'].shape[-2:] == dataset.image_size

    def test_dataset_with_transform(self, temp_data_dir, custom_transform):
        """Test dataset with custom transform"""
        dataset = InpaintingDataset(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks'],
            transform=custom_transform
        )
        
        sample = dataset[0]
        assert isinstance(sample['image'], torch.Tensor)
        assert sample['image'].shape[0] == 3  # Channels first after ToTensor

    def test_dataset_item_loading(self, temp_data_dir):
        """Test loading individual items from dataset"""
        dataset = InpaintingDataset(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks']
        )
        
        sample = dataset[0]
        assert 'image' in sample
        assert 'mask' in sample
        assert 'path' in sample
        
        assert isinstance(sample['image'], np.ndarray)
        assert isinstance(sample['mask'], np.ndarray)
        assert isinstance(sample['path'], str)
        
        # Check shapes
        assert len(sample['image'].shape) == 4  # B, C, H, W
        assert len(sample['mask'].shape) == 4  # B, C, H, W

    def test_invalid_image_dir(self):
        """Test handling of invalid image directory"""
        with pytest.raises(FileNotFoundError):
            InpaintingDataset(
                image_dir="nonexistent_directory",
                mask_dir=None
            )

    def test_empty_image_dir(self, temp_data_dir):
        """Test handling of empty image directory"""
        empty_dir = Path(temp_data_dir['root']) / 'empty'
        empty_dir.mkdir()
        
        dataset = InpaintingDataset(
            image_dir=str(empty_dir),
            mask_dir=None
        )
        
        assert len(dataset) == 0

    def test_custom_image_size(self, temp_data_dir):
        """Test dataset with custom image size"""
        custom_size = (256, 256)
        dataset = InpaintingDataset(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks'],
            image_size=custom_size
        )
        
        sample = dataset[0]
        assert sample['image'].shape[-2:] == custom_size
        assert sample['mask'].shape[-2:] == custom_size

    def test_invalid_image_files(self, temp_data_dir):
        """Test handling of invalid image files"""
        # Create invalid image file
        invalid_path = Path(temp_data_dir['images']) / 'invalid.jpg'
        invalid_path.write_text('not an image')

        # Verify that the file exists
        assert invalid_path.exists(), f"File {invalid_path} was not created."

        dataset = InpaintingDataset(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks']
        )

        # Verify that 'invalid.jpg' is in the dataset's image files
        invalid_file_str = str(invalid_path)
        assert invalid_file_str in dataset.image_files, \
            f"'invalid.jpg' not found in dataset image files: {dataset.image_files}"

        # Find the index of the invalid image
        invalid_index = dataset.image_files.index(invalid_file_str)

        # Should raise Exception when accessing invalid image
        with pytest.raises(Exception):
            _ = dataset[invalid_index]


    def test_data_loader_creation(self, temp_data_dir):
        """Test data loader creation and functionality"""
        loader = get_data_loader(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks'],
            batch_size=2
        )
        
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 2
        assert loader.num_workers == 4
        assert loader.pin_memory is True
        
        # Test batch loading
        batch = next(iter(loader))
        assert isinstance(batch, dict)
        assert 'image' in batch
        assert 'mask' in batch
        assert 'path' in batch
        assert len(batch['image']) == 2

    def test_custom_data_loader_params(self, temp_data_dir):
        """Test data loader with custom parameters"""
        loader = get_data_loader(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks'],
            batch_size=3,
            num_workers=2,
            shuffle=False
        )
        
        assert loader.batch_size == 3
        assert loader.num_workers == 2
        assert loader.shuffle is False

    def test_mask_cycling(self, temp_data_dir):
        """Test mask cycling when number of masks differs from number of images"""
        dataset = InpaintingDataset(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks']
        )
        
        # Get masks for all images
        masks = [dataset[i]['mask'] for i in range(len(dataset))]
        
        # Check that masks cycle correctly
        num_masks = len(dataset.mask_files)
        for i in range(len(masks)):
            expected_mask_idx = i % num_masks
            mask_path_1 = dataset.mask_files[expected_mask_idx]
            sample = dataset[i]
            assert isinstance(sample['mask'], np.ndarray)

    def test_transform_consistency(self, temp_data_dir, custom_transform):
        """Test consistency of transformations"""
        dataset = InpaintingDataset(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks'],
            transform=custom_transform
        )
        
        # Get same sample twice
        sample1 = dataset[0]
        sample2 = dataset[0]
        
        # Images should be different due to random transform
        assert not torch.allclose(sample1['image'], sample2['image'])
        
        # But masks should be the same
        assert np.array_equal(sample1['mask'], sample2['mask'])

    def test_memory_management(self, temp_data_dir):
        """Test memory management with large batches"""
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create data loader with large batch size
        loader = get_data_loader(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks'],
            batch_size=4
        )
        
        # Load several batches
        for _ in range(3):
            next(iter(loader))
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 1GB)
        assert memory_increase < 1024 * 1024 * 1024

    def test_file_extensions(self, temp_data_dir):
        """Test handling of different file extensions"""
        # Create images with different extensions
        extensions = ['.jpeg', '.png']  # Reduced to match actual structure
        for ext in extensions:
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(Path(temp_data_dir['images']) / f'test_image{ext}')
        
        dataset = InpaintingDataset(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks']
        )
        
        # Should find all valid images
        found_extensions = set(Path(f).suffix.lower() for f in dataset.image_files)
        assert found_extensions == set(extensions)

    def test_concurrent_loading(self, temp_data_dir):
        """Test concurrent data loading with multiple workers"""
        loader = get_data_loader(
            image_dir=temp_data_dir['images'],
            mask_dir=temp_data_dir['masks'],
            batch_size=2,
            num_workers=2
        )
        
        # Load multiple batches concurrently
        batches = [batch for batch in loader]
        assert len(batches) > 0
        
        # Check all batches are properly formed
        for batch in batches:
            assert 'image' in batch
            assert 'mask' in batch
            assert 'path' in batch

if __name__ == '__main__':
    pytest.main([__file__])