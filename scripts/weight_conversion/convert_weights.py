# scripts/weight_conversion/convert_weights.py
import sys
from pathlib import Path

# Add the root directory of the project to the Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import os
from pathlib import Path
import torch
from scripts.weight_conversion.converter import convert_pconv_weights, convert_vgg_weights

def load_nvidia_weights(weights_path):
    """Load and verify NVIDIA's pretrained weights"""
    try:
        # Load weights with CPU mapping
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        if 'state_dict' in checkpoint:
            print("Found valid NVIDIA weights with state_dict")
            return checkpoint['state_dict']
        else:
            print("Found valid NVIDIA weights")
            return checkpoint
    except Exception as e:
        print(f"Error loading NVIDIA weights: {e}")
        return None

def main():
    # Setup paths
    base_dir = Path("D:/Inpainting_tool2")
    weights_dir = base_dir / "weights" / "pconv"
    
    # Create directories if they don't exist
    (weights_dir / "unet").mkdir(parents=True, exist_ok=True)
    (weights_dir / "vgg16").mkdir(parents=True, exist_ok=True)
    
    # First check for NVIDIA weights
    nvidia_vgg_path = weights_dir / "vgg16" / "pdvgg16_bn_model_best.pth.tar"
    if nvidia_vgg_path.exists():
        print("Found NVIDIA VGG weights, using these instead of converting")
        vgg_weights = load_nvidia_weights(nvidia_vgg_path)
        if vgg_weights is not None:
            # Copy to our standard location if needed
            if str(nvidia_vgg_path) != str(weights_dir / "vgg16" / "vgg16_weights.pth"):
                torch.save(vgg_weights, weights_dir / "vgg16" / "vgg16_weights.pth")
                print("Copied NVIDIA weights to standard location")
    else:
        # Fall back to converting Keras weights
        vgg_h5_path = base_dir / "temp_weights" / "pytorch_to_keras_vgg16.h5"
        if vgg_h5_path.exists():
            print("No NVIDIA weights found, converting Keras VGG weights")
            convert_vgg_weights(str(vgg_h5_path), str(weights_dir / "vgg16" / "vgg16_weights.pth"))
    
    # Convert PConv weights if needed
    pconv_h5_path = base_dir / "temp_weights" / "pconv_imagenet.h5"
    if pconv_h5_path.exists():
        print("Converting PConv weights")
        convert_pconv_weights(str(pconv_h5_path), str(weights_dir / "unet" / "model_weights.pth"))

if __name__ == "__main__":
    main()