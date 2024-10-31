# scripts/weight_conversion/verify_weights.py

import os
import torch
import logging
from pathlib import Path
import h5py
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightVerifier:
    def __init__(self):
        self.base_dir = Path("D:/Inpainting_tool2")
        self.weights_dir = self.base_dir / "weights" / "pconv"
        
        # Expected layer names in our PyTorch model
        self.expected_unet_layers = {
            'enc1.0.weight', 'enc1.1.weight', 'enc1.1.bias',
            'enc2.0.weight', 'enc2.1.weight', 'enc2.1.bias',
            'enc3.0.weight', 'enc3.1.weight', 'enc3.1.bias',
            'enc4.0.weight', 'enc4.1.weight', 'enc4.1.bias',
            'dec1.0.weight', 'dec1.1.weight', 'dec1.1.bias',
            'dec2.0.weight', 'dec2.1.weight', 'dec2.1.bias',
            'dec3.0.weight', 'dec3.1.weight', 'dec3.1.bias',
            'dec4.0.weight', 'dec4.1.weight', 'dec4.1.bias',
        }
        
        self.expected_vgg_layers = {
            'features.0.weight', 'features.0.bias',
            'features.2.weight', 'features.2.bias',
            'features.5.weight', 'features.5.bias',
            'features.7.weight', 'features.7.bias',
        }

    def verify_unet_weights(self):
        """Verify UNet weights"""
        unet_path = self.weights_dir / "unet" / "model_weights.pth"
        
        try:
            weights = torch.load(unet_path, map_location='cpu')
            logger.info(f"\nVerifying UNet weights at: {unet_path}")
            print(f"\nVerifying UNet weights at: {unet_path}")
            logger.info(f"File size: {os.path.getsize(unet_path) / 1024:.2f} KB")
            print(f"File size: {os.path.getsize(unet_path) / 1024:.2f} KB")
            
            if not isinstance(weights, dict):
                logger.error("❌ UNet weights file is not a state dict!")
                print("❌ UNet weights file is not a state dict!")
                return False
                
            # Check contents
            logger.info("\nUNet Weights Analysis:")
            print("\nUNet Weights Analysis:")
            logger.info(f"Number of layers: {len(weights)}")
            print(f"Number of layers: {len(weights)}")
            
            # Check tensor shapes
            logger.info("\nLayer shapes:")
            print("\nLayer shapes:")
            for name, tensor in weights.items():
                logger.info(f"{name}: {tensor.shape}")
                print(f"{name}: {tensor.shape}")
            
            # Verify all layers exist
            missing_layers = self.expected_unet_layers - set(weights.keys())
            if missing_layers:
                logger.error(f"\n❌ Missing expected layers: {missing_layers}")
                print(f"\n❌ Missing expected layers: {missing_layers}")
                return False
                
            logger.info("\n✓ UNet weights verification passed")
            print("\n✓ UNet weights verification passed")
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Error verifying UNet weights: {e}")
            print(f"\n❌ Error verifying UNet weights: {e}")
            return False

    def verify_vgg_weights(self):
        """Verify VGG weights"""
        vgg_path = self.weights_dir / "vgg16" / "vgg16_weights.pth"
        
        try:
            weights = torch.load(vgg_path, map_location='cpu')
            logger.info(f"\nVerifying VGG weights at: {vgg_path}")
            print(f"\nVerifying VGG weights at: {vgg_path}")
            logger.info(f"File size: {os.path.getsize(vgg_path) / 1024:.2f} KB")
            print(f"File size: {os.path.getsize(vgg_path) / 1024:.2f} KB")
            
            if not isinstance(weights, dict):
                logger.error("❌ VGG weights file is not a state dict!")
                print("❌ VGG weights file is not a state dict!")
                return False
                
            # Check contents
            logger.info("\nVGG Weights Analysis:")
            print("\nVGG Weights Analysis:")
            logger.info(f"Number of layers: {len(weights)}")
            print(f"Number of layers: {len(weights)}")
            
            # Check tensor shapes
            logger.info("\nLayer shapes:")
            print("\nLayer shapes:")
            for name, tensor in weights.items():
                logger.info(f"{name}: {tensor.shape}")
                print(f"{name}: {tensor.shape}")
            
            # Verify important layers exist
            missing_layers = self.expected_vgg_layers - set(weights.keys())
            if missing_layers:
                logger.error(f"\n❌ Missing expected layers: {missing_layers}")
                print(f"\n❌ Missing expected layers: {missing_layers}")
                return False
                
            logger.info("\n✓ VGG weights verification passed")
            print("\n✓ VGG weights verification passed")
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Error verifying VGG weights: {e}")
            print(f"\n❌ Error verifying VGG weights: {e}")
            return False

    def verify_original_h5(self):
        """Verify original Keras H5 file"""
        h5_path = self.base_dir / "temp_weights" / "pconv_imagenet.h5"
        
        try:
            with h5py.File(h5_path, 'r') as f:
                logger.info(f"\nVerifying original H5 file at: {h5_path}")
                print(f"\nVerifying original H5 file at: {h5_path}")
                logger.info(f"File size: {os.path.getsize(h5_path) / 1024:.2f} KB")
                print(f"File size: {os.path.getsize(h5_path) / 1024:.2f} KB")
                
                # Print model structure
                logger.info("\nH5 Model Structure:")
                print("\nH5 Model Structure:")
                def print_structure(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        logger.info(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
                        print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
                
                f.visititems(print_structure)
                
            logger.info("\n✓ H5 file verification passed")
            print("\n✓ H5 file verification passed")
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Error verifying H5 file: {e}")
            print(f"\n❌ Error verifying H5 file: {e}")
            return False

    def verify_all(self):
        """Run all verifications"""
        h5_ok = self.verify_original_h5()
        unet_ok = self.verify_unet_weights()
        vgg_ok = self.verify_vgg_weights()
        
        logger.info("\n=== Summary ===")
        print("\n=== Summary ===")
        logger.info(f"Original H5: {'✓' if h5_ok else '❌'}")
        print(f"Original H5: {'✓' if h5_ok else '❌'}")
        logger.info(f"UNet weights: {'✓' if unet_ok else '❌'}")
        print(f"UNet weights: {'✓' if unet_ok else '❌'}")
        logger.info(f"VGG weights: {'✓' if vgg_ok else '❌'}")
        print(f"UNet weights: {'✓' if unet_ok else '❌'}")
        
        return all([h5_ok, unet_ok, vgg_ok])

if __name__ == "__main__":
    verifier = WeightVerifier()
    verifier.verify_all()