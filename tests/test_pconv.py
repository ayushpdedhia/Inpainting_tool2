# D:\Inpainting_tool2\tests\test_pconv.py
# cd D:\Inpainting_tool2
# python -m pytest tests/test_pconv.py     
import pytest
import torch
import numpy as np
from src.models.pconv.layers.partialconv2d import PartialConv2d
from src.models.pconv.loss import PConvLoss
from src.models.pconv.vgg_extractor import VGG16FeatureExtractor
import torch.nn as nn  # For nn.BatchNorm2d, nn.Upsample
from src.models.pconv.models.pconv_unet import PConvUNet  # For PConvUNet model
from src.models.pconv.vgg_extractor import VGG16FeatureExtractor, gram_matrix

class TestPartialConv2d:
    
    @pytest.fixture
    def device(self, request):
        """Return CPU device for testing"""
        return torch.device('cpu')

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
        conv_layer.eval()  # Set to evaluation mode
        
        # Create sample input with batch size 2
        input_tensor = torch.randn(2, 3, 64, 64).to(device)
        mask = torch.ones(2, 1, 64, 64).to(device)
        
        # Test without mask
        with torch.no_grad():
            output_no_mask = conv_layer(input_tensor)
        assert isinstance(output_no_mask, torch.Tensor)
        assert output_no_mask.shape == (2, 64, 64, 64)
        
        # Test with mask
        with torch.no_grad():
            output_with_mask, updated_mask = conv_layer(input_tensor, mask)
        assert output_with_mask.shape == (2, 64, 64, 64)
        assert updated_mask.shape == (2, 1, 64, 64)
    
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
    def device(self, request):
        """Return CPU device for testing"""
        return torch.device('cpu')
    
    @pytest.fixture
    def loss_function(self, device):
        """Initialize PConv loss function"""
        return PConvLoss(device=device)
    
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

    def test_loss_weights(self):
        """Test loss weight scaling"""
        device = torch.device('cpu')
        custom_weights = {
            'l1_weight': 2.0,
            'hole_weight': 12.0,
            'perceptual_weight': 0.1,
            'style_weight': 240.0,
            'tv_weight': 0.2,
            'device': device
        }
        
        loss_fn = PConvLoss(**custom_weights)
        
        # Test if weights are properly applied
        output = torch.randn(1, 3, 64, 64, device=device)
        target = torch.randn(1, 3, 64, 64, device=device)
        mask = torch.ones(1, 1, 64, 64, device=device)
        
        _, loss_dict = loss_fn(output, target, mask)
        
        # Check if weights are properly applied
        assert torch.allclose(
            loss_dict['valid'], 
            custom_weights['l1_weight'] * torch.mean(torch.abs(mask * (output - target)))
        )

    def test_gradients(self):
        """Test gradient computation"""
        device = torch.device('cpu')
        loss_fn = PConvLoss(device=device)
        output = torch.randn(1, 3, 64, 64, requires_grad=True, device=device)
        target = torch.randn(1, 3, 64, 64, device=device)
        mask = torch.ones(1, 1, 64, 64, device=device)
        
        loss, _ = loss_fn(output, target, mask)
        loss.backward()
        assert output.grad is not None

class TestVGGFeatureExtractor:
    @pytest.fixture
    def device(self, request):
        """Return CPU device for testing"""
        return torch.device('cpu')

    @pytest.fixture
    def feature_extractor(self, device):
        """Initialize VGG feature extractor"""
        return VGG16FeatureExtractor().to(device)
    
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

    def test_gram_matrix(self, device):
        """Test Gram matrix computation"""
        input_tensor = torch.randn(2, 3, 64, 64).to(device)
        gram = gram_matrix(input_tensor)
        assert gram.shape == (2, 3, 3)
        # Test symmetry
        assert torch.allclose(gram, gram.transpose(-2, -1))
        # Test positive semi-definiteness
        eigenvalues = torch.linalg.eigvalsh(gram[0])
        assert torch.all(eigenvalues >= -1e-6)  # Allow for numerical error

    def test_gram_matrix_batch(self, device):
        """Test Gram matrix computation with different batch sizes"""
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 64, 64).to(device)
            gram = gram_matrix(input_tensor)
            assert gram.shape == (batch_size, 3, 3)

    def test_gram_matrix_values(self, device):
        """Test gram matrix with simpler input for clearer verification"""
        # Create a simpler test case
        input_tensor = torch.zeros(1, 2, 2, 2).to(device)
        input_tensor[0, 0] = torch.tensor([[2., 2.], [2., 2.]]).to(device)  # Constant values
        input_tensor[0, 1] = torch.tensor([[3., 3.], [3., 3.]]).to(device)  # Constant values
        
        gram = gram_matrix(input_tensor)
        
        # Calculate expected values (simpler now with constant inputs)
        expected_00 = 16.0  # 4 * 2 * 2 
        expected_11 = 36.0  # 4 * 3 * 3
        expected_01 = 24.0  # 4 * 2 * 3
        
        assert torch.allclose(gram[0, 0, 0], torch.tensor(expected_00).to(device), rtol=1e-4)
        assert torch.allclose(gram[0, 1, 1], torch.tensor(expected_11).to(device), rtol=1e-4)
        assert torch.allclose(gram[0, 0, 1], torch.tensor(expected_01).to(device), rtol=1e-4)

    def test_layer_selection(self):
        """Test layer selection behavior"""
        extractor = VGG16FeatureExtractor(layer_num=2)
        x = torch.randn(1, 3, 256, 256)
        features = extractor(x)
        assert len(features) == 2

    def test_normalization(self):
        """Test normalization with controlled input"""
        # Create input with known statistics
        x = torch.ones(1, 3, 64, 64) * 0.5  # Constant value for predictable results
        normalized = VGG16FeatureExtractor.normalize_batch(x)

        # Check shape preserved
        assert normalized.shape == x.shape

        # Expected means after ImageNet normalization
        expected_means = (0.5 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
        channel_means = normalized.mean(dim=(2, 3))
        channel_stds = normalized.std(dim=(2, 3))

        # Allow for some numerical error
        assert torch.allclose(channel_means, expected_means.view(1, -1), atol=1e-4), f"Means not as expected: {channel_means}"


class TestPConvUNet:
    @pytest.fixture
    def device(self, request):
        """Return CPU device for testing"""
        return torch.device('cpu')
    
    @pytest.fixture
    def model(self):
        """Initialize PConv UNet model"""
        return PConvUNet(input_channels=3)
        
    def test_initialization(self, model):
        """Test model initialization"""
        assert isinstance(model, PConvUNet)
        # Test encoder blocks
        assert isinstance(model.enc1[0], PartialConv2d)
        assert isinstance(model.enc1[1], nn.BatchNorm2d)
        # Test decoder blocks
        assert isinstance(model.dec1[0], PartialConv2d)
        assert isinstance(model.dec1[1], nn.BatchNorm2d)
    
    def test_forward_pass(self, model, device):
        model = model.to(device)
        model.eval()
        
        # Use batch size of 2 for forward pass
        x = torch.randn(2, 3, 256, 256).to(device)
        mask = torch.ones(2, 1, 256, 256).to(device)
        
        with torch.no_grad():
            output = model(x, mask)
        assert output.shape == x.shape
    
    def test_skip_connections(self, model, device):
        """Test skip connections"""
        model = model.to(device)
        x = torch.randn(1, 3, 256, 256).to(device)
        mask = torch.ones(1, 1, 256, 256).to(device)
        
        # Access intermediate features
        with torch.no_grad():
            enc1, m1 = model.enc1(x, mask)
            assert enc1.shape == (1, 64, 128, 128)
    

    def test_mask_propagation(self, model, device):
        model = model.to(device)
        model.eval()
        
        # Use batch size of 2
        x = torch.randn(2, 3, 256, 256).to(device)
        mask = torch.ones(2, 1, 256, 256).to(device)
        mask[:, :, 100:150, 100:150] = 0
        
        with torch.no_grad():
            output = model(x, mask)
        
        # Test hole region modification
        orig_hole = x[:, :, 100:150, 100:150]
        new_hole = output[:, :, 100:150, 100:150]
        assert not torch.allclose(orig_hole, new_hole)
    

    def test_upsampling(self, model, device):
        """Test upsampling with high contrast pattern and batch size > 1"""
        torch.manual_seed(0)  # For reproducibility
        
        # Create sharper contrast pattern
        x = torch.zeros(2, 3, 32, 32, device=device)
        x[:, :, :16, :16] = 1.0   # Top-left quadrant
        x[:, :, 16:, 16:] = -1.0  # Bottom-right quadrant
        
        # Create a mask with some holes
        mask = torch.ones(2, 1, 32, 32, device=device)
        mask[:, :, 8:24, 8:24] = 0  # Create hole in the middle
        
        # Models with different upsampling modes
        model_nearest = PConvUNet(upsampling_mode='nearest').to(device)
        model_bilinear = PConvUNet(upsampling_mode='bilinear').to(device)
        
        # Ensure models are in eval mode for consistent behavior
        model_nearest.eval()
        model_bilinear.eval()
        
        with torch.no_grad():
            out_nearest = model_nearest(x, mask)
            out_bilinear = model_bilinear(x, mask)
        
        # Calculate difference at the edges where upsampling modes differ most
        edge_mask = torch.zeros_like(x)
        edge_mask[:, :, 15:17, :] = 1  # Horizontal edge
        edge_mask[:, :, :, 15:17] = 1  # Vertical edge
        
        edge_diff = ((out_nearest - out_bilinear) * edge_mask).abs().mean().item()
        
        assert edge_diff > 1e-3, f"Upsampling modes should differ at edges (diff={edge_diff})"
