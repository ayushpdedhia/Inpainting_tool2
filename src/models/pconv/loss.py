import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models  # Added this import

# Import the feature extractor class defined above
from .vgg_extractor import VGG16FeatureExtractor, gram_matrix

class PConvLoss(nn.Module):
    """
    Partial Convolution Loss Module combining both implementations
    """
    def __init__(self, 
                 l1_weight=1.0,
                 hole_weight=6.0,
                 perceptual_weight=0.05,
                 style_weight=120.0,
                 tv_weight=0.1,
                 feat_num=4,
                 device="cuda"):
        super().__init__()
        
        # Check if CUDA is available, if not use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vgg = VGG16FeatureExtractor(layer_num=feat_num).to(device)
        self.l1_loss = nn.L1Loss()
        
        # Loss weights
        self.l1_weight = l1_weight
        self.hole_weight = hole_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        self.device = device

    def total_variation_loss(self, x, mask):
        """Total variation loss with dilated mask"""
        kernel = torch.ones(3, 3).to(self.device)
        dilated_mask = F.conv2d(1 - mask, 
                               kernel.unsqueeze(0).unsqueeze(0),
                               padding=1)
        dilated_mask = torch.clamp(dilated_mask, 0, 1)
        
        loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]) * dilated_mask[:, :, :, :-1]) + \
               torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]) * dilated_mask[:, :, :-1, :])
        return loss

    def forward(self, output, target, mask):
        """
        Calculate total PConv loss
        
        Args:
            output: Model output [B, C, H, W]
            target: Ground truth [B, C, H, W]
            mask: Binary mask [B, 1, H, W]
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary containing individual loss components
        """
        # Valid area and hole losses
        l_valid = torch.mean(torch.abs(mask * (output - target)))
        l_hole = torch.mean(torch.abs((1 - mask) * (output - target)))
        
        # Composite output
        comp = output * (1 - mask) + target * mask
        
        # Get normalized VGG features
        output_feats = self.vgg(output)
        with torch.no_grad():
            target_feats = self.vgg(target)
        comp_feats = self.vgg(comp)
        
        # Perceptual loss
        l_perceptual = 0
        for out_feat, comp_feat, target_feat in zip(output_feats, comp_feats, target_feats):
            l_perceptual += self.l1_loss(out_feat, target_feat)
            l_perceptual += self.l1_loss(comp_feat, target_feat)
            
        # Style loss with efficient Gram matrix
        l_style = 0
        for out_feat, comp_feat, target_feat in zip(output_feats, comp_feats, target_feats):
            target_gram = gram_matrix(target_feat)
            out_gram = gram_matrix(out_feat)
            comp_gram = gram_matrix(comp_feat)
            
            l_style += self.l1_loss(out_gram, target_gram)
            l_style += self.l1_loss(comp_gram, target_gram)
            
        # Total variation loss
        l_tv = self.total_variation_loss(comp, mask)
        
        # Weighted total loss
        loss = (self.l1_weight * l_valid + 
                self.hole_weight * l_hole + 
                self.perceptual_weight * l_perceptual + 
                self.style_weight * l_style + 
                self.tv_weight * l_tv)
        
        # Return individual losses for logging
        loss_dict = {
            'total': loss,
            'valid': self.l1_weight * l_valid,
            'hole': self.hole_weight * l_hole,
            'perceptual': self.perceptual_weight * l_perceptual,
            'style': self.style_weight * l_style,
            'tv': self.tv_weight * l_tv
        }
        
        return loss, loss_dict