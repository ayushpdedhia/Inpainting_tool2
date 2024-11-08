import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.partialconv2d import PartialConv2d

class EncoderBlock(nn.Module):
    """Encoder block with partial convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False):
        super().__init__()
        self.conv = PartialConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias, return_mask=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def __getitem__(self, idx):
        if idx == 0:
            return self.conv
        elif idx == 1:
            return self.bn
        elif idx == 2:
            return self.relu
        raise IndexError("Index out of range")

    def forward(self, x, mask_in):
        """
        mask_in: 1 = keep original, 0 = hole/inpaint
        """
        x, mask = self.conv(x, mask_in)
        if self.training:
            x = self.bn(x)
        x = self.relu(x)
        return x, mask

class DecoderBlock(nn.Module):
    """Decoder block with partial convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = PartialConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=1, padding=padding, return_mask=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def __getitem__(self, idx):
        if idx == 0:
            return self.conv
        elif idx == 1:
            return self.bn
        raise IndexError("Index out of range")

    def forward(self, x, mask_in):
        """
        mask_in: 1 = keep original, 0 = hole/inpaint
        """
        x, mask = self.conv(x, mask_in)
        if self.training:
            x = self.bn(x)
        x = self.leaky_relu(x)
        return x, mask

class PConvUNet(nn.Module):
    """
    PConv-UNet for image inpainting.
    Mask convention throughout:
    1 = valid/keep original pixels
    0 = holes/pixels to inpaint
    """
    def __init__(self, input_channels=3, layer_size=7, upsampling_mode='nearest'):
        super().__init__()
        self.upsampling_mode = upsampling_mode

        # Encoder path (with mask propagation)
        self.enc1 = EncoderBlock(input_channels, 64, 7, stride=2, padding=3)
        self.enc2 = EncoderBlock(64, 128, 5)
        self.enc3 = EncoderBlock(128, 256, 5)
        self.enc4 = EncoderBlock(256, 512, 3)
        self.enc5 = EncoderBlock(512, 512, 3)
        self.enc6 = EncoderBlock(512, 512, 3)
        self.enc7 = EncoderBlock(512, 512, 3)
        self.enc8 = EncoderBlock(512, 512, 3)

        # Decoder path
        self.dec8 = DecoderBlock(512 + 512, 512)
        self.dec7 = DecoderBlock(512 + 512, 512)
        self.dec6 = DecoderBlock(512 + 512, 512)
        self.dec5 = DecoderBlock(512 + 512, 512)
        self.dec4 = DecoderBlock(512 + 256, 256)
        self.dec3 = DecoderBlock(256 + 128, 128)
        self.dec2 = DecoderBlock(128 + 64, 64)
        self.dec1 = DecoderBlock(64 + input_channels, input_channels)

        self.final = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upsample(self, x, shape=None):
        """
        Upsample tensor to target shape or by factor of 2
        """
        if shape is not None:
            return F.interpolate(
                x, size=shape, mode=self.upsampling_mode,
                align_corners=True if self.upsampling_mode == 'bilinear' else None
            )
        return F.interpolate(
            x, scale_factor=2, mode=self.upsampling_mode,
            align_corners=True if self.upsampling_mode == 'bilinear' else None
        )

    def forward(self, x, mask):
        """
        Forward pass of PConv-UNet
        Args:
            x: Input image [B, C, H, W]
            mask: Binary mask [B, 1, H, W] where:
                  1 = valid pixels (keep original)
                  0 = holes (to be filled)
        """
        training = self.training
        try:
            # Input validation
            print("\n=== PConv-UNet Input Stats ===")
            print(f"Image - Shape: {x.shape}, Range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"Mask - Shape: {mask.shape}, Range: [{mask.min():.3f}, {mask.max():.3f}]")
            print(f"Mask unique values: {torch.unique(mask).cpu().numpy()}")

            if mask.shape[1] > 1:
                mask = mask[:, :1]

            # Encoder path
            print("\n=== Encoder Path ===")
            enc1, m1 = self.enc1(x, mask)
            self._debug_features("E1", enc1, m1)

            enc2, m2 = self.enc2(enc1, m1)
            self._debug_features("E2", enc2, m2)

            enc3, m3 = self.enc3(enc2, m2)
            self._debug_features("E3", enc3, m3)

            enc4, m4 = self.enc4(enc3, m3)
            self._debug_features("E4", enc4, m4)

            enc5, m5 = self.enc5(enc4, m4)
            self._debug_features("E5", enc5, m5)

            enc6, m6 = self.enc6(enc5, m5)
            self._debug_features("E6", enc6, m6)

            enc7, m7 = self.enc7(enc6, m6)
            self._debug_features("E7", enc7, m7)

            enc8, m8 = self.enc8(enc7, m7)
            self._debug_features("E8", enc8, m8)

            # Decoder path
            print("\n=== Decoder Path ===")

            # Handler for each decoder level
            def decoder_level(level, enc_feat, enc_mask, prev_feat, prev_mask, dec_block):
                # Upsample previous features and mask
                up_feat = self.upsample(prev_feat, shape=enc_feat.shape[2:])
                up_mask = self.upsample(prev_mask, shape=enc_feat.shape[2:])
                
                # Combine masks from encoder and previous decoder layer
                cat_mask = torch.cat([up_mask, enc_mask], dim=1).mean(dim=1, keepdim=True)
                
                # Concatenate features for skip connection
                cat_feat = torch.cat([up_feat, enc_feat], dim=1)
                
                # Apply decoder block
                dec_feat, dec_mask = dec_block(cat_feat, cat_mask)
                self._debug_features(f"D{level}", dec_feat, dec_mask)
                
                return dec_feat, dec_mask
            
            # Decoder levels
            dec8, dm8 = decoder_level(8, enc7, m7, enc8, m8, self.dec8)
            dec7, dm7 = decoder_level(7, enc6, m6, dec8, dm8, self.dec7)
            dec6, dm6 = decoder_level(6, enc5, m5, dec7, dm7, self.dec6)
            dec5, dm5 = decoder_level(5, enc4, m4, dec6, dm6, self.dec5)
            dec4, dm4 = decoder_level(4, enc3, m3, dec5, dm5, self.dec4)
            dec3, dm3 = decoder_level(3, enc2, m2, dec4, dm4, self.dec3)
            dec2, dm2 = decoder_level(2, enc1, m1, dec3, dm3, self.dec2)
            dec1, dm1 = decoder_level(1, x, mask, dec2, dm2, self.dec1)

            # Final output
            output = self.final(torch.cat([dec1, x], 1))
            output = torch.clamp(output, 0.0, 1.0)

            print("\n=== Final Composition ===")
            print(f"Generated output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"Final mask unique values: {torch.unique(mask).cpu().numpy()}")

            # Final composition: use mask directly (1=keep original, 0=use generated)
            final_output = x * mask + output * (1 - mask)

            print(f"Composition stats:")
            print(f"Overall range: [{final_output.min():.3f}, {final_output.max():.3f}]")
            print(f"Original regions: [{(final_output * mask).min():.3f}, {(final_output * mask).max():.3f}]")
            print(f"Generated regions: [{(final_output * (1-mask)).min():.3f}, {(final_output * (1-mask)).max():.3f}]")
            
            return final_output

        finally:
            # Restore original training state
            self.train(training)

    def _debug_features(self, name, features, mask):
        """Debug helper to print feature and mask statistics"""
        print(f"\n{name} stats:")
        print(f"Features - Shape: {features.shape}, Range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"Mask - Shape: {mask.shape}, Range: [{mask.min():.3f}, {mask.max():.3f}]")