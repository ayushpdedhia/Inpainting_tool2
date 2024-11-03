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
        # Handle no-mask case first
        if mask_in is None:
            # Important: Return only convolution output for no-mask case
            return super(PartialConv2d, self).forward(input)
        
        assert len(input.shape) == 4
        
        # Create mask if none provided
        if mask_in is None:
            mask = torch.ones(input.data.shape[0], 1, 
                            input.data.shape[2], input.data.shape[3]).to(input)
        else:
            mask = mask_in
            
        # Update mask and compute mask ratio
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, 
                                        bias=None, stride=self.stride, 
                                        padding=self.padding, dilation=self.dilation, 
                                        groups=1)

                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # Apply convolution to masked input
        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask))

        # Apply bias if present
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        # Return output with or without mask
        if self.return_mask:
            if self.multi_channel:
                self.update_mask = self.update_mask.mean(dim=1, keepdim=True)
            return output, self.update_mask
        else:
            return output