import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.partialconv2d import PartialConv2d

class EncoderBlock(nn.Module):
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
        x, mask = self.conv(x, mask_in)
        if self.training:
            x = self.bn(x)
        x = self.relu(x)
        return x, mask

class DecoderBlock(nn.Module):
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
        x, mask = self.conv(x, mask_in)
        if self.training:
            x = self.bn(x)
        x = self.leaky_relu(x)
        return x, mask

class PConvUNet(nn.Module):
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
            nn.ReLU(inplace=True)  # Changed from Sigmoid to ReLU to preserve color range
        )

    def upsample(self, x):
        """
        Upsample the input tensor by a factor of 2
        Modified to ensure more distinct differences between modes
        """
        if self.upsampling_mode == 'bilinear':
            # Use bilinear with align_corners=True for smoother interpolation
            return F.interpolate(
                x,
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            )
        else:
            # Nearest neighbor will preserve sharp edges
            return F.interpolate(
                x,
                scale_factor=2,
                mode='nearest'
            )

    def forward(self, x, mask):
        """
        Args:
            x: Input image [B, C, H, W]
            mask: Binary mask [B, 1, H, W]
        """
        training = self.training
        try:
            if mask.shape[1] > 1:
                mask = mask[:, :1]

            # Encoder path
            enc1, m1 = self.enc1(x, mask)
            enc2, m2 = self.enc2(enc1, m1)
            enc3, m3 = self.enc3(enc2, m2)
            enc4, m4 = self.enc4(enc3, m3)
            enc5, m5 = self.enc5(enc4, m4)
            enc6, m6 = self.enc6(enc5, m5)
            enc7, m7 = self.enc7(enc6, m6)
            enc8, m8 = self.enc8(enc7, m7)

            # Decoder path with skip connections and dimension checks
            # Level 8
            up8 = self.upsample(enc8)
            up8_mask = self.upsample(m8)
            if up8.size(2) != enc7.size(2) or up8.size(3) != enc7.size(3):
                up8 = F.interpolate(up8, size=(enc7.size(2), enc7.size(3)),
                                  mode=self.upsampling_mode,
                                  align_corners=True if self.upsampling_mode == 'bilinear' else None)
                up8_mask = F.interpolate(up8_mask, size=(enc7.size(2), enc7.size(3)),
                                       mode=self.upsampling_mode,
                                       align_corners=True if self.upsampling_mode == 'bilinear' else None)
            cat_mask8 = torch.cat([up8_mask, m7], dim=1).mean(dim=1, keepdim=True)
            dec8, dm8 = self.dec8(torch.cat([up8, enc7], 1), cat_mask8)

            # Level 7
            up7 = self.upsample(dec8)
            up7_mask = self.upsample(dm8)
            if up7.size(2) != enc6.size(2) or up7.size(3) != enc6.size(3):
                up7 = F.interpolate(up7, size=(enc6.size(2), enc6.size(3)),
                                  mode=self.upsampling_mode,
                                  align_corners=True if self.upsampling_mode == 'bilinear' else None)
                up7_mask = F.interpolate(up7_mask, size=(enc6.size(2), enc6.size(3)),
                                       mode=self.upsampling_mode,
                                       align_corners=True if self.upsampling_mode == 'bilinear' else None)
            cat_mask7 = torch.cat([up7_mask, m6], dim=1).mean(dim=1, keepdim=True)
            dec7, dm7 = self.dec7(torch.cat([up7, enc6], 1), cat_mask7)

            # Level 6
            up6 = self.upsample(dec7)
            up6_mask = self.upsample(dm7)
            if up6.size(2) != enc5.size(2) or up6.size(3) != enc5.size(3):
                up6 = F.interpolate(up6, size=(enc5.size(2), enc5.size(3)),
                                  mode=self.upsampling_mode,
                                  align_corners=True if self.upsampling_mode == 'bilinear' else None)
                up6_mask = F.interpolate(up6_mask, size=(enc5.size(2), enc5.size(3)),
                                       mode=self.upsampling_mode,
                                       align_corners=True if self.upsampling_mode == 'bilinear' else None)
            cat_mask6 = torch.cat([up6_mask, m5], dim=1).mean(dim=1, keepdim=True)
            dec6, dm6 = self.dec6(torch.cat([up6, enc5], 1), cat_mask6)

            # Level 5
            up5 = self.upsample(dec6)
            up5_mask = self.upsample(dm6)
            if up5.size(2) != enc4.size(2) or up5.size(3) != enc4.size(3):
                up5 = F.interpolate(up5, size=(enc4.size(2), enc4.size(3)),
                                  mode=self.upsampling_mode,
                                  align_corners=True if self.upsampling_mode == 'bilinear' else None)
                up5_mask = F.interpolate(up5_mask, size=(enc4.size(2), enc4.size(3)),
                                       mode=self.upsampling_mode,
                                       align_corners=True if self.upsampling_mode == 'bilinear' else None)
            cat_mask5 = torch.cat([up5_mask, m4], dim=1).mean(dim=1, keepdim=True)
            dec5, dm5 = self.dec5(torch.cat([up5, enc4], 1), cat_mask5)

            # Level 4
            up4 = self.upsample(dec5)
            up4_mask = self.upsample(dm5)
            if up4.size(2) != enc3.size(2) or up4.size(3) != enc3.size(3):
                up4 = F.interpolate(up4, size=(enc3.size(2), enc3.size(3)),
                                  mode=self.upsampling_mode,
                                  align_corners=True if self.upsampling_mode == 'bilinear' else None)
                up4_mask = F.interpolate(up4_mask, size=(enc3.size(2), enc3.size(3)),
                                       mode=self.upsampling_mode,
                                       align_corners=True if self.upsampling_mode == 'bilinear' else None)
            cat_mask4 = torch.cat([up4_mask, m3], dim=1).mean(dim=1, keepdim=True)
            dec4, dm4 = self.dec4(torch.cat([up4, enc3], 1), cat_mask4)

            # Level 3
            up3 = self.upsample(dec4)
            up3_mask = self.upsample(dm4)
            if up3.size(2) != enc2.size(2) or up3.size(3) != enc2.size(3):
                up3 = F.interpolate(up3, size=(enc2.size(2), enc2.size(3)),
                                  mode=self.upsampling_mode,
                                  align_corners=True if self.upsampling_mode == 'bilinear' else None)
                up3_mask = F.interpolate(up3_mask, size=(enc2.size(2), enc2.size(3)),
                                       mode=self.upsampling_mode,
                                       align_corners=True if self.upsampling_mode == 'bilinear' else None)
            cat_mask3 = torch.cat([up3_mask, m2], dim=1).mean(dim=1, keepdim=True)
            dec3, dm3 = self.dec3(torch.cat([up3, enc2], 1), cat_mask3)

            # Level 2
            up2 = self.upsample(dec3)
            up2_mask = self.upsample(dm3)
            if up2.size(2) != enc1.size(2) or up2.size(3) != enc1.size(3):
                up2 = F.interpolate(up2, size=(enc1.size(2), enc1.size(3)),
                                  mode=self.upsampling_mode,
                                  align_corners=True if self.upsampling_mode == 'bilinear' else None)
                up2_mask = F.interpolate(up2_mask, size=(enc1.size(2), enc1.size(3)),
                                       mode=self.upsampling_mode,
                                       align_corners=True if self.upsampling_mode == 'bilinear' else None)
            cat_mask2 = torch.cat([up2_mask, m1], dim=1).mean(dim=1, keepdim=True)
            dec2, dm2 = self.dec2(torch.cat([up2, enc1], 1), cat_mask2)

            # Level 1
            up1 = self.upsample(dec2)
            up1_mask = self.upsample(dm2)
            if up1.size(2) != x.size(2) or up1.size(3) != x.size(3):
                up1 = F.interpolate(up1, size=(x.size(2), x.size(3)),
                                  mode=self.upsampling_mode,
                                  align_corners=True if self.upsampling_mode == 'bilinear' else None)
                up1_mask = F.interpolate(up1_mask, size=(x.size(2), x.size(3)),
                                       mode=self.upsampling_mode,
                                       align_corners=True if self.upsampling_mode == 'bilinear' else None)
            cat_mask1 = torch.cat([up1_mask, mask], dim=1).mean(dim=1, keepdim=True)
            dec1, dm1 = self.dec1(torch.cat([up1, x], 1), cat_mask1)

            # Final output
            output = self.final(torch.cat([dec1, x], 1))

            # Before returning, add clamping
            output = self.final(torch.cat([dec1, x], 1))
            output = torch.clamp(output, 0.0, 1.0)  # Clamp to valid range

            # Combine with original image using mask
            return output * (1 - mask) + x * mask

        finally:
            # Restore original training state
            self.train(training)