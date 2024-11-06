# tests/test_weight_loader.py
# python -m pytest tests/test_weight_loader.py 

import pytest
import torch
import torch.nn as nn
import os
import yaml
import tempfile
import shutil
from pathlib import Path
from collections import OrderedDict
from src.utils.weight_loader import WeightLoader

class MockModel(nn.Module):
    """Mock model for testing weight loading"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.vgg = self._create_mock_vgg()
        
    def _create_mock_vgg(self):
        vgg = nn.Module()
        vgg.features = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True)
        )
        return vgg
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return x

class TestWeightLoader:
    """Test suite for weight loading functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Fixture providing temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Fixture providing mock configuration"""
        config = {
            'paths': {
                'unet_weights': os.path.join(temp_dir, 'unet_weights.pth'),
                'vgg_weights': os.path.join(temp_dir, 'vgg_weights.pth')
            }
        }
        return config

    @pytest.fixture
    def config_file(self, temp_dir, mock_config):
        """Fixture providing temporary config file"""
        config_path = os.path.join(temp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(mock_config, f)
        return config_path

    @pytest.fixture
    def mock_state_dict(self, weight_loader):
        """Fixture providing mock state dictionary"""
        return OrderedDict([
            ('conv1.weight', torch.randn(64, 3, 3, 3).to(weight_loader.device)),
            ('conv1.bias', torch.randn(64).to(weight_loader.device)),
            ('bn1.weight', torch.randn(64).to(weight_loader.device)),
            ('bn1.bias', torch.randn(64).to(weight_loader.device)),
            ('bn1.running_mean', torch.randn(64).to(weight_loader.device)),
            ('bn1.running_var', torch.randn(64).to(weight_loader.device))
        ])

    @pytest.fixture
    def mock_vgg_state_dict(self, weight_loader):
        """Fixture providing mock VGG state dictionary"""
        return OrderedDict([
            ('features.0.weight', torch.randn(64, 3, 3, 3).to(weight_loader.device)),
            ('features.0.bias', torch.randn(64).to(weight_loader.device)),
            ('features.2.weight', torch.randn(64, 64, 3, 3).to(weight_loader.device)),
            ('features.2.bias', torch.randn(64).to(weight_loader.device))
        ])

    @pytest.fixture
    def weight_loader(self, config_file):
        """Fixture providing WeightLoader instance"""
        return WeightLoader(config_file)

    def test_initialization(self, config_file):
        """Test WeightLoader initialization"""
        loader = WeightLoader(config_file)
        assert loader.config is not None
        assert isinstance(loader.device, torch.device)
        assert loader.device.type in ['cuda', 'cpu']

    def test_config_loading(self, config_file, mock_config):
        """Test configuration loading"""
        loader = WeightLoader(config_file)
        assert loader.config == mock_config
        assert 'paths' in loader.config
        assert 'unet_weights' in loader.config['paths']
        assert 'vgg_weights' in loader.config['paths']

    def test_invalid_config_path(self):
        """Test handling of invalid config path"""
        with pytest.raises(FileNotFoundError):
            WeightLoader('nonexistent_config.yaml')

    def test_device_selection(self, weight_loader):
        """Test device selection logic"""
        device = weight_loader.get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'cpu']
        assert device == weight_loader.device

    def test_checkpoint_logging(self, weight_loader, tmp_path):
        """Test checkpoint information logging"""
        checkpoint = {
            'epoch': 10,
            'best_prec1': 95.5,
            'date': '2024-03-01'
        }
        
        # Create a dummy checkpoint file
        checkpoint_path = tmp_path / "checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Test logging function
        weight_loader._log_checkpoint_info(checkpoint, checkpoint_path)
        # Note: We can't directly test logger output, but we can verify the function runs

    def test_load_model_weights(self, weight_loader, mock_state_dict, tmp_path):
        """Test loading model weights"""
        model = MockModel()
        initial_state = model.state_dict()
        
        # Save mock weights
        weights_path = tmp_path / "weights.pth"
        torch.save(mock_state_dict, weights_path)
        
        # Update config to use temporary weights path
        weight_loader.config['paths']['unet_weights'] = str(weights_path)
        
        # Load weights
        model = weight_loader.load_model_weights(model, load_vgg=False)
        
        # Verify weights were loaded
        for key in mock_state_dict.keys():
            assert torch.equal(model.state_dict()[key], mock_state_dict[key])

    def test_load_vgg_weights(self, weight_loader, mock_vgg_state_dict, tmp_path):
        """Test loading VGG weights"""
        model = MockModel()
        
        # Save mock VGG weights
        vgg_path = tmp_path / "vgg_weights.pth"
        torch.save(mock_vgg_state_dict, vgg_path)
        
        # Update config
        weight_loader.config['paths']['vgg_weights'] = str(vgg_path)
        
        # Load weights
        model = weight_loader.load_model_weights(model, load_vgg=True)
        
        # Verify VGG weights were loaded
        for key in mock_vgg_state_dict.keys():
            assert torch.equal(model.vgg.state_dict()[key], mock_vgg_state_dict[key])

    def test_dataparallel_handling(self, weight_loader, mock_state_dict, tmp_path):
        """Test handling of DataParallel wrapped weights"""
        # Create DataParallel style state dict
        dp_state_dict = OrderedDict([
            (f'module.{k}', v) for k, v in mock_state_dict.items()
        ])
        
        # Save weights
        weights_path = tmp_path / "dp_weights.pth"
        torch.save(dp_state_dict, weights_path)
        
        # Update config
        weight_loader.config['paths']['unet_weights'] = str(weights_path)
        
        # Load weights
        model = MockModel()
        model = weight_loader.load_model_weights(model, load_vgg=False)
        
        # Verify weights were properly loaded without 'module' prefix
        for key in mock_state_dict.keys():
            assert torch.equal(model.state_dict()[key], mock_state_dict[key])

    def test_missing_weight_files(self, weight_loader):
        """Test handling of missing weight files"""
        model = MockModel()
        
        # Update config with nonexistent paths
        weight_loader.config['paths']['unet_weights'] = 'nonexistent.pth'
        weight_loader.config['paths']['vgg_weights'] = 'nonexistent_vgg.pth'
        
        # Should not raise error, but log warning
        model = weight_loader.load_model_weights(model)
        assert isinstance(model, MockModel)

    def test_corrupt_weight_file(self, weight_loader, tmp_path):
        """Test handling of corrupt weight files"""
        # Create corrupt weight file
        corrupt_path = tmp_path / "corrupt_weights.pth"
        with open(corrupt_path, 'w') as f:
            f.write("not a valid weight file")
        
        # Update config
        weight_loader.config['paths']['unet_weights'] = str(corrupt_path)
        
        model = MockModel()
        with pytest.raises(Exception):
            weight_loader.load_model_weights(model)

    def test_device_mapping(self, weight_loader, mock_state_dict, tmp_path):
        """Test weight loading with different devices"""
        model = MockModel().to(weight_loader.device)
        
        # Save weights
        weights_path = tmp_path / "weights.pth"
        torch.save(mock_state_dict, weights_path)
        weight_loader.config['paths']['unet_weights'] = str(weights_path)
        
        # Load weights
        model = weight_loader.load_model_weights(model)
        
        # Verify model is on correct device
        model_device = next(model.parameters()).device
        assert model_device.type == weight_loader.device.type
        if weight_loader.device.type == 'cuda':
            assert model_device.index == weight_loader.device.index

    def test_partial_state_dict(self, weight_loader, mock_state_dict, tmp_path):
        """Test loading partial state dict"""
        # Create partial state dict
        partial_dict = OrderedDict([
            (k, v) for i, (k, v) in enumerate(mock_state_dict.items()) 
            if i < len(mock_state_dict) // 2
        ])
        
        # Save partial weights
        weights_path = tmp_path / "partial_weights.pth"
        torch.save(partial_dict, weights_path)
        weight_loader.config['paths']['unet_weights'] = str(weights_path)
        
        model = MockModel()
        # Should load without error due to strict=False
        model = weight_loader.load_model_weights(model)
        assert isinstance(model, MockModel)

    def test_memory_efficiency(self, weight_loader, mock_state_dict, tmp_path):
        """Test memory efficiency during weight loading"""
        import psutil
        process = psutil.Process()
        
        # Save large weights
        weights_path = tmp_path / "large_weights.pth"
        large_dict = OrderedDict([
            (f'layer_{i}.weight', torch.randn(100, 100))
            for i in range(100)
        ])
        torch.save(large_dict, weights_path)
        weight_loader.config['paths']['unet_weights'] = str(weights_path)
        
        # Measure memory usage
        initial_memory = process.memory_info().rss
        model = MockModel()
        model = weight_loader.load_model_weights(model)
        final_memory = process.memory_info().rss
        
        memory_increase = final_memory - initial_memory
        # Memory increase should be reasonable (less than 1GB for this test)
        assert memory_increase < 1024 * 1024 * 1024

if __name__ == '__main__':
    pytest.main([__file__])