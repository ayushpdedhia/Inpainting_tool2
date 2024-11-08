# test_vgg_extraction.py

import torch
import cv2
import numpy as np
from PIL import Image
from src.models.pconv.vgg_extractor import VGG16FeatureExtractor
from src.utils.image_processor import ImageProcessor
from src.models.pconv.models.pconv_unet import PConvUNet
from pathlib import Path

def test_vgg_features():
    # Initialize components
    vgg = VGG16FeatureExtractor(layer_num=4).cuda()
    image_processor = ImageProcessor()
    model = PConvUNet().cuda()
    model.eval()

    # Load test image
    test_image_path = "data/test_samples/images/test_image_001.jpeg"  # Use your test image path
    image = Image.open(test_image_path)
    
    # Create simple mask (circle in center)
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(mask, (256, 256), 100, 255, -1)
    
    print("=== Stage 1: Initial Input ===")
    print(f"Image size: {image.size}")
    print(f"Mask shape: {mask.shape}")

    # Preprocess
    processed_image, processed_mask = image_processor.preprocess(image, mask)
    
    print("\n=== Stage 2: After Preprocessing ===")
    print(f"Processed image shape: {processed_image.shape}")
    print(f"Processed image range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    print(f"Processed mask shape: {processed_mask.shape}")
    print(f"Processed mask range: [{processed_mask.min():.3f}, {processed_mask.max():.3f}]")

    # Convert to tensors and move to GPU
    image_tensor = torch.from_numpy(processed_image).cuda()
    mask_tensor = torch.from_numpy(processed_mask).cuda()

    # Extract VGG features
    with torch.no_grad():
        features = vgg(image_tensor)
        
        print("\n=== Stage 3: VGG Feature Maps ===")
        for idx, feat in enumerate(features):
            print(f"Feature level {idx + 1}:")
            print(f"  Shape: {feat.shape}")
            print(f"  Range: [{feat.min().item():.3f}, {feat.max().item():.3f}]")
            print(f"  Mean: {feat.mean().item():.3f}")
            print(f"  Std: {feat.std().item():.3f}")

        # Run through PConv-UNet and track mask propagation
        print("\n=== Stage 4: Mask Propagation Through Encoder ===")
        
        # Get encoder outputs
        enc1, m1 = model.enc1(image_tensor, mask_tensor)
        enc2, m2 = model.enc2(enc1, m1)
        enc3, m3 = model.enc3(enc2, m2)
        enc4, m4 = model.enc4(enc3, m3)
        enc5, m5 = model.enc5(enc4, m4)
        enc6, m6 = model.enc6(enc5, m5)
        enc7, m7 = model.enc7(enc6, m6)
        enc8, m8 = model.enc8(enc7, m7)

        # Print encoder stats
        encoder_outputs = [(enc1, m1), (enc2, m2), (enc3, m3), (enc4, m4),
                         (enc5, m5), (enc6, m6), (enc7, m7), (enc8, m8)]
        
        for idx, (feat, mask) in enumerate(encoder_outputs):
            print(f"\nEncoder block {idx + 1}:")
            print(f"  Feature shape: {feat.shape}")
            print(f"  Feature range: [{feat.min().item():.3f}, {feat.max().item():.3f}]")
            print(f"  Feature mean: {feat.mean().item():.3f}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask unique values: {torch.unique(mask).cpu().numpy()}")
            print(f"  Mask mean: {mask.mean().item():.3f}")

        # Run full forward pass and get final composition
        print("\n=== Stage 5: Final Composition ===")
        output = model(image_tensor, mask_tensor)
        
        print("Final output stats:")
        print(f"  Shape: {output.shape}")
        print(f"  Range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print(f"  Mean: {output.mean().item():.3f}")
        
        # Save intermediate visualizations
        output_dir = Path("debug_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Convert tensors back to images and save
        def save_tensor_as_image(tensor, name):
            img = tensor.cpu().numpy()[0].transpose(1, 2, 0)
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        save_tensor_as_image(image_tensor, "input")
        save_tensor_as_image(output, "output")
        
        # Save masks
        for idx, (_, mask) in enumerate(encoder_outputs):
            mask_np = mask.cpu().numpy()[0, 0]
            mask_np = (mask_np * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"mask_level_{idx+1}.png"), mask_np)

if __name__ == "__main__":
    test_vgg_features()