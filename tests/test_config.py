# D:\Inpainting_tool2\tests\test_config.py
# cd D:\Inpainting_tool2
# python -m pytest tests/test_config.py

import pytest
import os
import yaml
import tempfile
from pathlib import Path
import shutil

class TestConfiguration:
    """Test suite for configuration validation and loading"""

    @pytest.fixture
    def base_config(self):
        """Fixture providing a base valid configuration"""
        return {
            'model': {
                'name': 'pconv_unet',
                'weights_dir': 'weights/pconv',
                'input_size': [512, 512],
                'device': 'cuda'
            },
            'paths': {
                'data_dir': 'data',
                'weights_dir': 'weights',
                'temp_weights': 'temp_weights',
                'unet_weights': 'weights/pconv/unet/model_weights.pth',
                'vgg_weights': 'weights/pconv/vgg16/vgg16_weights.pth'
            },
            'interface': {
                'canvas_size': 512,
                'max_image_size': 1024,
                'supported_formats': ['jpg', 'jpeg', 'png']
            }
        }

    @pytest.fixture
    def temp_config_dir(self):
        """Fixture providing a temporary directory for config files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def create_config_file(self, config_data, temp_dir):
        """Helper method to create a config file"""
        config_path = Path(temp_dir) / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        return config_path

    def test_valid_config_loading(self, base_config, temp_config_dir):
        """Test loading a valid configuration file"""
        config_path = self.create_config_file(base_config, temp_config_dir)
        
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config == base_config
        assert 'model' in loaded_config
        assert 'paths' in loaded_config
        assert 'interface' in loaded_config

    def test_model_configuration(self, base_config, temp_config_dir):
        """Test model-specific configuration validation"""
        # Test valid model config
        config_path = self.create_config_file(base_config, temp_config_dir)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        assert config['model']['name'] == 'pconv_unet'
        assert isinstance(config['model']['input_size'], list)
        assert len(config['model']['input_size']) == 2
        assert config['model']['device'] in ['cuda', 'cpu']

    def test_missing_required_fields(self, base_config, temp_config_dir):
        """Test handling of missing required configuration fields"""
        # Remove required fields
        del base_config['model']['name']
        
        config_path = self.create_config_file(base_config, temp_config_dir)
        
        with pytest.raises(KeyError):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                assert config['model']['name']  # Should raise KeyError

    def test_invalid_model_name(self, base_config, temp_config_dir):
        """Test handling of invalid model name"""
        base_config['model']['name'] = 'invalid_model'
        config_path = self.create_config_file(base_config, temp_config_dir)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            assert config['model']['name'] != 'pconv_unet'

    def test_path_validation(self, base_config, temp_config_dir):
        """Test validation of path configurations"""
        # Test relative paths
        config_path = self.create_config_file(base_config, temp_config_dir)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        for key, path in config['paths'].items():
            assert isinstance(path, str)
            assert not os.path.isabs(path)  # Ensure relative paths

    def test_interface_configuration(self, base_config, temp_config_dir):
        """Test interface configuration validation"""
        config_path = self.create_config_file(base_config, temp_config_dir)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        assert isinstance(config['interface']['canvas_size'], int)
        assert isinstance(config['interface']['max_image_size'], int)
        assert isinstance(config['interface']['supported_formats'], list)

    def test_invalid_device_configuration(self, base_config, temp_config_dir):
        """Test handling of invalid device configuration"""
        base_config['model']['device'] = 'invalid_device'
        config_path = self.create_config_file(base_config, temp_config_dir)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            assert config['model']['device'] not in ['cuda', 'cpu']

    def test_invalid_input_size(self, base_config, temp_config_dir):
        """Test handling of invalid input size configuration"""
        # Test with non-list input size
        base_config['model']['input_size'] = 512
        config_path = self.create_config_file(base_config, temp_config_dir)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            assert not isinstance(config['model']['input_size'], list)

        # Test with invalid list length
        base_config['model']['input_size'] = [512]
        config_path = self.create_config_file(base_config, temp_config_dir)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            assert len(config['model']['input_size']) != 2

    def test_supported_formats_validation(self, base_config, temp_config_dir):
        """Test validation of supported image formats"""
        # Test with invalid format
        base_config['interface']['supported_formats'].append('invalid_format')
        config_path = self.create_config_file(base_config, temp_config_dir)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            assert 'invalid_format' in config['interface']['supported_formats']
            assert not all(fmt in ['jpg', 'jpeg', 'png'] 
                         for fmt in config['interface']['supported_formats'])

    def test_weight_paths_validation(self, base_config, temp_config_dir):
        """Test validation of weight file paths"""
        # Test with invalid weight paths
        base_config['paths']['unet_weights'] = 'invalid/path/weights.pth'
        config_path = self.create_config_file(base_config, temp_config_dir)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            assert not os.path.exists(config['paths']['unet_weights'])

    def test_config_type_validation(self, base_config, temp_config_dir):
        """Test validation of configuration value types"""
        # Test canvas size type
        base_config['interface']['canvas_size'] = '512'  # Should be int
        config_path = self.create_config_file(base_config, temp_config_dir)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            assert not isinstance(config['interface']['canvas_size'], int)

    def test_environment_specific_config(self, base_config, temp_config_dir):
        """Test environment-specific configuration handling"""
        # Test CUDA availability affecting device configuration
        import torch
        cuda_available = torch.cuda.is_available()
        
        if not cuda_available:
            base_config['model']['device'] = 'cpu'
            
        config_path = self.create_config_file(base_config, temp_config_dir)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
            if not cuda_available:
                assert config['model']['device'] == 'cpu'

    def test_config_file_permissions(self, base_config, temp_config_dir):
        """Test configuration file permission handling"""
        config_path = self.create_config_file(base_config, temp_config_dir)
        
        # Test read permissions
        assert os.access(config_path, os.R_OK)
        
        # Test write permissions
        assert os.access(config_path, os.W_OK)

    def test_malformed_yaml(self, temp_config_dir):
        """Test handling of malformed YAML configuration"""
        # Create malformed YAML
        malformed_yaml = """
        model:
            name: pconv_unet
            input_size: [512, 512
            device: cuda
        """
        
        config_path = Path(temp_config_dir) / 'malformed_config.yaml'
        with open(config_path, 'w') as f:
            f.write(malformed_yaml)
        
        with pytest.raises(yaml.YAMLError):
            with open(config_path, 'r') as f:
                yaml.safe_load(f)

if __name__ == '__main__':
    pytest.main([__file__])