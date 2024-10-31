import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.partialconv2d import PartialConv2d

class PConvUNet(nn.Module):
    def __init__(self, input_channels=3, layer_size=7, upsampling_mode='nearest'):
        super().__init__()
        
        # Encoder
        def encoder_block(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False):
            return nn.Sequential(
                PartialConv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, padding=padding, bias=bias, return_mask=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # Decoder
        def decoder_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                PartialConv2d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, return_mask=True),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )

        # Encoder path (with mask propagation)
        self.enc1 = encoder_block(input_channels, 64, 7, stride=2, padding=3)
        self.enc2 = encoder_block(64, 128, 5)
        self.enc3 = encoder_block(128, 256, 5)
        self.enc4 = encoder_block(256, 512, 3)
        self.enc5 = encoder_block(512, 512, 3)
        self.enc6 = encoder_block(512, 512, 3)
        self.enc7 = encoder_block(512, 512, 3)
        self.enc8 = encoder_block(512, 512, 3)

        # Decoder path
        self.up = nn.Upsample(scale_factor=2, mode=upsampling_mode)
        self.dec8 = decoder_block(512 + 512, 512)
        self.dec7 = decoder_block(512 + 512, 512)
        self.dec6 = decoder_block(512 + 512, 512)
        self.dec5 = decoder_block(512 + 512, 512)
        self.dec4 = decoder_block(512 + 256, 256)
        self.dec3 = decoder_block(256 + 128, 128)
        self.dec2 = decoder_block(128 + 64, 64)
        self.dec1 = decoder_block(64 + input_channels, input_channels, kernel_size=3, padding=1)

        self.final = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        """
        Args:
            x: Input image [B, C, H, W]
            mask: Binary mask [B, 1, H, W]
        """
        # Encoder
        enc1, m1 = self.enc1(x, mask)
        enc2, m2 = self.enc2(enc1, m1)
        enc3, m3 = self.enc3(enc2, m2)
        enc4, m4 = self.enc4(enc3, m3)
        enc5, m5 = self.enc5(enc4, m4)
        enc6, m6 = self.enc6(enc5, m5)
        enc7, m7 = self.enc7(enc6, m6)
        enc8, m8 = self.enc8(enc7, m7)

        # Decoder with skip connections
        up8 = self.up(enc8)
        up8_mask = self.up(m8)
        dec8, dm8 = self.dec8(torch.cat([up8, enc7], 1), torch.cat([up8_mask, m7], 1))

        up7 = self.up(dec8)
        up7_mask = self.up(dm8)
        dec7, dm7 = self.dec7(torch.cat([up7, enc6], 1), torch.cat([up7_mask, m6], 1))

        up6 = self.up(dec7)
        up6_mask = self.up(dm7)
        dec6, dm6 = self.dec6(torch.cat([up6, enc5], 1), torch.cat([up6_mask, m5], 1))

        up5 = self.up(dec6)
        up5_mask = self.up(dm6)
        dec5, dm5 = self.dec5(torch.cat([up5, enc4], 1), torch.cat([up5_mask, m4], 1))

        up4 = self.up(dec5)
        up4_mask = self.up(dm5)
        dec4, dm4 = self.dec4(torch.cat([up4, enc3], 1), torch.cat([up4_mask, m3], 1))

        up3 = self.up(dec4)
        up3_mask = self.up(dm4)
        dec3, dm3 = self.dec3(torch.cat([up3, enc2], 1), torch.cat([up3_mask, m2], 1))

        up2 = self.up(dec3)
        up2_mask = self.up(dm3)
        dec2, dm2 = self.dec2(torch.cat([up2, enc1], 1), torch.cat([up2_mask, m1], 1))

        up1 = self.up(dec2)
        up1_mask = self.up(dm2)
        dec1, dm1 = self.dec1(torch.cat([up1, x], 1), torch.cat([up1_mask, mask], 1))

        # Final output
        output = self.final(torch.cat([dec1, x], 1))
        
        # Combine with original image using mask
        return output * (1 - mask) + x * mask