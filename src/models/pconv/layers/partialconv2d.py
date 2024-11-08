###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # Extract custom arguments before calling super()
        multi_channel = kwargs.pop('multi_channel', False) if 'multi_channel' in kwargs else False
        return_mask = kwargs.pop('return_mask', True) if 'return_mask' in kwargs else True
        
        # Call parent constructor
        super(PartialConv2d, self).__init__(*args, **kwargs)
        
        # Store the arguments as instance variables
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        
        # Initialize weight_maskUpdater AFTER super() call
        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels, self.in_channels, 
                self.kernel_size[0], self.kernel_size[1]
            )
        else:
            self.weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1]
            )
        
        # Calculate the sliding window size
        self.slide_winsize = (self.weight_maskUpdater.shape[1] * 
                            self.weight_maskUpdater.shape[2] * 
                            self.weight_maskUpdater.shape[3])
        
        # Register buffer for optimization (changed name to avoid conflict)
        self.register_buffer('weight_mask_updater', self.weight_maskUpdater)
        
        # Initialize other attributes
        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        """
        Forward pass of Partial Convolution.
        
        Mask Convention:
        - mask_in: 1 = keep original pixels, 0 = holes to be filled
        - internal: work directly with the mask (no inversions needed)
        - output: maintain same convention as input
        
        Args:
            input: Input tensor [B, C, H, W]
            mask_in: Binary mask tensor [B, 1, H, W] where:
                    1 = valid pixels (keep original)
                    0 = holes (to be filled)
        """

        # Debug input stats
        print("\n=== PConv Layer Input Stats ===")
        print(f"Input tensor - Shape: {input.shape}")
        print(f"Input range: [{input.min():.3f}, {input.max():.3f}]")
        if mask_in is not None:
            print(f"Input mask - Shape: {mask_in.shape}")
            print(f"Mask range: [{mask_in.min():.3f}, {mask_in.max():.3f}]")
            print(f"Mask unique values: {torch.unique(mask_in)}")
        
        # Handle no-mask case first
        if mask_in is None:
            return super(PartialConv2d, self).forward(input)
        
        assert len(input.shape) == 4, "Input should be a 4D tensor"

        # Use input mask directly - no need to invert
        mask = mask_in.clone()  # Clone to avoid modifying original mask
        

        # Update mask and compute mask ratio only if needed
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)
            print(f"PConv input mask unique values: {torch.unique(mask_in)}")

            with torch.no_grad():
                # Ensure weight_maskUpdater is on correct device
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                # Compute update mask directly from input mask
                # No inversion needed - process valid regions directly
                self.update_mask = F.conv2d(mask,
                                          self.weight_maskUpdater,
                                          bias=None,
                                          stride=self.stride,
                                          padding=self.padding,
                                          dilation=self.dilation)

                # Update mask ratio for proper scaling
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

                # Debug updated mask
                print("\n=== Mask Update Stats ===")
                print(f"Update mask range: [{self.update_mask.min():.3f}, {self.update_mask.max():.3f}]")
                print(f"Update mask unique values: {torch.unique(self.update_mask)}")
                print(f"Mask ratio range: [{self.mask_ratio.min():.3f}, {self.mask_ratio.max():.3f}]")

            # Apply convolution to valid regions (where mask = 1)
            masked_input = torch.mul(input, mask)
            raw_out = super(PartialConv2d, self).forward(masked_input)

            # Enhanced feature computation
            if self.bias is not None:
                bias_view = self.bias.view(1, self.out_channels, 1, 1)
                # Compute features
                hole_features = raw_out * self.mask_ratio  # Apply mask ratio to all features
                valid_features = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
                
                # Use torch.where for cleaner feature selection
                output = torch.where(
                    self.update_mask > 0.5,  # Threshold for valid regions
                    valid_features,  # Use valid features where mask is 1
                    hole_features  # Use hole features where mask is 0
                )
            else:
                # Simplified unbiased case
                output = raw_out * self.mask_ratio  # Apply mask ratio uniformly

            # Add gradual transition for boundary regions
            edge_mask = F.max_pool2d(1 - self.update_mask, 3, stride=1, padding=1)
            boundary_mask = edge_mask - (1 - self.update_mask)
            output = output * (1 - boundary_mask) + raw_out * boundary_mask

            # Debug output stats
            print("\n=== Layer Output Stats ===")
            print(f"Output tensor range: [{output.min():.3f}, {output.max():.3f}]")

            # Return output and mask
            if self.return_mask:
                if self.multi_channel:
                    final_mask = self.update_mask.mean(dim=1, keepdim=True)
                else:
                    final_mask = self.update_mask
                print(f"Output mask unique values: {torch.unique(final_mask)}")
                # Return mask directly - maintain input convention
                return output, final_mask
            else:
                return output