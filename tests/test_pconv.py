import pytest
import torch
import numpy as np
from src.models.pconv.layers.partialconv2d import PartialConv2d
from src.models.pconv.loss import PConvLoss
from src.models.pconv.vgg_extractor import VGG16FeatureExtractor

class TestPartialConv2d:
    @pytest.fixture
    def conv_layer(self):
        """Initialize a basic partial convolution layer"""
        return PartialConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
    
    @pytest.fixture
    def device(self):
        """Get the appropriate device"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_initialization(self, conv_layer):
        """Test layer initialization"""
        assert isinstance(conv_layer, PartialConv2d)
        assert conv_layer.in_channels == 3
        assert conv_layer.out_channels == 64
        assert conv_layer.kernel_size == (3, 3)
        
        # Check weight initialization
        assert conv_layer.weight_maskUpdater is not None
        assert conv_layer.weight_maskUpdater.shape == (1, 1, 3, 3)
    
    def test_forward_pass(self, conv_layer, device):
        """Test forward pass with and without mask"""
        conv_layer = conv_layer.to(device)
        
        # Create sample input
        input_tensor = torch.randn(1, 3, 64, 64).to(device)
        mask = torch.ones(1, 1, 64, 64).to(device)
        
        # Test without mask
        output_no_mask = conv_layer(input_tensor)
        assert output_no_mask.shape == (1, 64, 64, 64)
        
        # Test with mask
        output_with_mask, updated_mask = conv_layer(input_tensor, mask)
        assert output_with_mask.shape == (1, 64, 64, 64)
        assert updated_mask.shape == (1, 1, 64, 64)
    
    def test_mask_update(self, conv_layer, device):
        """Test mask update mechanism"""
        conv_layer = conv_layer.to(device)
        
        # Create input with hole
        input_tensor = torch.randn(1, 3, 64, 64).to(device)
        mask = torch.ones(1, 1, 64, 64).to(device)
        mask[:, :, 16:32, 16:32] = 0  # Create hole
        
        # Forward pass
        _, updated_mask = conv_layer(input_tensor, mask)
        
        # Check if mask was updated properly
        assert torch.any(updated_mask[:, :, 16:32, 16:32] < 1)  # Hole region
        assert torch.all(updated_mask >= 0) and torch.all(updated_mask <= 1)
    
    def test_multi_channel_mask(self, device):
        """Test with multi-channel mask"""
        conv_layer = PartialConv2d(
            3, 64, 3, 1, 1, 
            multi_channel=True
        ).to(device)
        
        input_tensor = torch.randn(1, 3, 64, 64).to(device)
        mask = torch.ones(1, 3, 64, 64).to(device)
        
        output, updated_mask = conv_layer(input_tensor, mask)
        assert output.shape == (1, 64, 64, 64)
        assert updated_mask.shape == (1, 1, 64, 64)

class TestPConvLoss:
    @pytest.fixture
    def loss_function(self, device):
        """Initialize PConv loss function"""
        return PConvLoss(device=device)
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_initialization(self, loss_function):
        """Test loss function initialization"""
        assert isinstance(loss_function.vgg, VGG16FeatureExtractor)
        assert loss_function.l1_loss is not None
    
    def test_loss_computation(self, loss_function, device):
        """Test full loss computation"""
        # Create sample data
        output = torch.randn(1, 3, 256, 256).to(device)
        target = torch.randn(1, 3, 256, 256).to(device)
        mask = torch.ones(1, 1, 256, 256).to(device)
        mask[:, :, 64:128, 64:128] = 0  # Create hole
        
        # Compute loss
        total_loss, loss_dict = loss_function(output, target, mask)
        
        # Check loss components
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_dict, dict)
        assert 'total' in loss_dict
        assert 'valid' in loss_dict
        assert 'hole' in loss_dict
        assert 'perceptual' in loss_dict
        assert 'style' in loss_dict
        assert 'tv' in loss_dict
        
        # Check loss values
        assert not torch.isnan(total_loss)
        assert all(not torch.isnan(v) for v in loss_dict.values())
    
    def test_valid_hole_loss(self, loss_function, device):
        """Test valid and hole region loss computation"""
        output = torch.ones(1, 3, 64, 64).to(device)
        target = torch.ones(1, 3, 64, 64).to(device)
        mask = torch.ones(1, 1, 64, 64).to(device)
        
        # Create difference in hole region
        mask[:, :, 16:32, 16:32] = 0
        output[:, :, 16:32, 16:32] = 0
        
        total_loss, loss_dict = loss_function(output, target, mask)
        
        # Hole loss should be higher than valid loss
        assert loss_dict['hole'] > loss_dict['valid']
    
    def test_perceptual_loss(self, loss_function, device):
        """Test perceptual loss computation"""
        output = torch.randn(1, 3, 256, 256).to(device)
        target = output.clone()
        mask = torch.ones(1, 1, 256, 256).to(device)
        
        # Modify small region
        output[:, :, 100:120, 100:120] += 0.5
        
        _, loss_dict = loss_function(output, target, mask)
        assert loss_dict['perceptual'] > 0
    
    def test_style_loss(self, loss_function, device):
        """Test style loss computation"""
        output = torch.randn(1, 3, 256, 256).to(device)
        target = output.clone()
        mask = torch.ones(1, 1, 256, 256).to(device)
        
        # Change style by adjusting color distribution
        output = output * 1.5
        
        _, loss_dict = loss_function(output, target, mask)
        assert loss_dict['style'] > 0
    
    def test_total_variation_loss(self, loss_function, device):
        """Test total variation loss"""
        output = torch.randn(1, 3, 64, 64).to(device)
        target = torch.randn(1, 3, 64, 64).to(device)
        mask = torch.ones(1, 1, 64, 64).to(device)
        mask[:, :, 16:32, 16:32] = 0
        
        # Create high-frequency pattern
        output[:, :, ::2, ::2] = 1
        output[:, :, 1::2, 1::2] = -1
        
        _, loss_dict = loss_function(output, target, mask)
        assert loss_dict['tv'] > 0

class TestVGGFeatureExtractor:
    @pytest.fixture
    def feature_extractor(self, device):
        """Initialize VGG feature extractor"""
        return VGG16FeatureExtractor().to(device)
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_initialization(self, feature_extractor):
        """Test VGG initialization"""
        assert isinstance(feature_extractor, VGG16FeatureExtractor)
        
        # Check if weights are frozen
        for param in feature_extractor.parameters():
            assert not param.requires_grad
    
    def test_feature_extraction(self, feature_extractor, device):
        """Test feature extraction"""
        input_tensor = torch.randn(1, 3, 256, 256).to(device)
        features = feature_extractor(input_tensor)
        
        assert isinstance(features, list)
        assert len(features) == 4  # Default 4 feature levels
        
        # Check feature map dimensions
        expected_sizes = [(64, 128, 128), (128, 64, 64),
                         (256, 32, 32), (512, 16, 16)]
        for feat, (c, h, w) in zip(features, expected_sizes):
            assert feat.shape[1:] == (c, h, w)