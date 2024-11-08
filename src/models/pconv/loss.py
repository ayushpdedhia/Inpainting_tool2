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
                 l1_weight=6.0, # Increase from 1.0
                 hole_weight=6.0,
                 perceptual_weight=0.05,
                 style_weight=5.0,  # Reduce from 60.0 to 5.0 
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

        print("\n=== PConvLoss Forward Pass ===")
        print(f"Input tensors shapes:")
        print(f"Output: {output.shape}, Range: {output.min():.3f} to {output.max():.3f}")
        print(f"Target: {target.shape}, Range: {target.min():.3f} to {target.max():.3f}")
        print(f"Mask: {mask.shape}, Range: {mask.min():.3f} to {mask.max():.3f}")

        # Add edge awareness to loss computation
        edge_mask = F.max_pool2d(1 - mask, 3, stride=1, padding=1)
        boundary_regions = edge_mask - (1 - mask)

        # Modify valid and hole losses to focus more on boundaries
        l_valid = torch.mean(torch.abs(mask * (output - target))) + \
                2.0 * torch.mean(torch.abs(boundary_regions * (output - target)))
        
        l_hole = torch.mean(torch.abs((1 - mask) * (output - target)))
        print(f"\nInitial Losses:")
        print(f"Valid Loss: {l_valid.item():.6f}")
        print(f"Hole Loss: {l_hole.item():.6f}")
        
        # Modify composition to use smooth transition
        alpha = F.sigmoid(mask * 5)  # Smoother transition
        comp = output * (1 - alpha) + target * alpha
        print(f"\nComposite Output:")
        print(f"Shape: {comp.shape}, Range: {comp.min():.3f} to {comp.max():.3f}")
        
        # Get normalized VGG features
        output_feats = self.vgg(output)
        with torch.no_grad():
            target_feats = self.vgg(target)
        comp_feats = self.vgg(comp)

        print(f"\n[Loss Debug] VGG feature stats:")
        print(f"Number of feature layers: {len(output_feats)}")
        for idx, (out_f, targ_f, comp_f) in enumerate(zip(output_feats, target_feats, comp_feats)):
            print(f"\nLayer {idx+1} stats:")
            print(f"Output features - Shape: {out_f.shape}, Mean: {out_f.mean():.3f}")
            print(f"Target features - Shape: {targ_f.shape}, Mean: {targ_f.mean():.3f}")
            print(f"Comp features - Shape: {comp_f.shape}, Mean: {comp_f.mean():.3f}")
        
        # Perceptual loss
        l_perceptual = 0
        print("\nCalculating Perceptual Loss:")
        for i, (out_feat, comp_feat, target_feat) in enumerate(zip(output_feats, comp_feats, target_feats)):
            out_loss = self.l1_loss(out_feat, target_feat)
            comp_loss = self.l1_loss(comp_feat, target_feat)
            l_perceptual += (out_loss + comp_loss)
            print(f"Layer {i+1} - Out Loss: {out_loss.item():.6f}, Comp Loss: {comp_loss.item():.6f}")
        
        # Style loss
        l_style = 0
        print("\nCalculating Style Loss:")
        for i, (out_feat, comp_feat, target_feat) in enumerate(zip(output_feats, comp_feats, target_feats)):
            target_gram = gram_matrix(target_feat)
            out_gram = gram_matrix(out_feat)
            comp_gram = gram_matrix(comp_feat)
            
            out_style_loss = self.l1_loss(out_gram, target_gram)
            comp_style_loss = self.l1_loss(comp_gram, target_gram)
            l_style += (out_style_loss + comp_style_loss)
            print(f"Layer {i+1} - Out Style Loss: {out_style_loss.item():.6f}, Comp Style Loss: {comp_style_loss.item():.6f}")
        
        # Total variation loss
        l_tv = self.total_variation_loss(comp, mask)
        print(f"\nTotal Variation Loss: {l_tv.item():.6f}")
        
        # Final weighted loss
        loss = (self.l1_weight * l_valid + 
                self.hole_weight * l_hole + 
                self.perceptual_weight * l_perceptual + 
                self.style_weight * l_style + 
                self.tv_weight * l_tv)
        
        # Compile loss dict
        loss_dict = {
            'total': loss,
            'valid': self.l1_weight * l_valid,
            'hole': self.hole_weight * l_hole,
            'perceptual': self.perceptual_weight * l_perceptual,
            'style': self.style_weight * l_style,
            'tv': self.tv_weight * l_tv
        }
        
        print("\n=== Final Loss Values ===")
        for key, value in loss_dict.items():
            print(f"{key.capitalize()}: {value.item():.6f}")
        
        return loss, loss_dict