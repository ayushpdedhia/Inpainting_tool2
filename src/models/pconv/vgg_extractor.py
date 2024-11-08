import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def gram_matrix(input_tensor):
    """Compute Gram matrix exactly as defined in the style transfer paper"""
    print(f"\n=== Gram Matrix Computation ===")
    print(f"Input tensor:")
    print(f"Shape: {input_tensor.shape}")
    print(f"Range: {input_tensor.min():.3f} to {input_tensor.max():.3f}")
    print(f"Mean: {input_tensor.mean():.3f}")
    
    b, ch, h, w = input_tensor.size()
    features = input_tensor.view(b, ch, h * w)
    print(f"\nReshaped features:")
    print(f"Shape: {features.shape}")
    
    features_t = features.transpose(1, 2)
    gram = torch.bmm(features, features_t)
    
    print(f"\nGram matrix:")
    print(f"Shape: {gram.shape}")
    print(f"Range: {gram.min():.3f} to {gram.max():.3f}")
    print(f"Mean: {gram.mean():.3f}")
    
    return gram

class VGG16FeatureExtractor(nn.Module):
    """VGG16 feature extraction with proper layer slicing"""
    def __init__(self, layer_num=4):
        super().__init__()
        vgg_pretrained = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        vgg_pretrained_features = vgg_pretrained.features
        
        self.layer_num = layer_num
        
        # Extract feature layers using NVIDIA's approach
        self.slice1 = nn.Sequential()
        for x in range(5):  # relu1_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        self.slice2 = nn.Sequential()
        for x in range(5, 10):  # relu2_2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        self.slice3 = nn.Sequential()
        for x in range(10, 17):  # relu3_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        self.slice4 = nn.Sequential()
        for x in range(17, 24):  # relu4_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    @staticmethod
    def normalize_batch(batch, div_factor=255.0):
        """Normalize batch for VGG processing"""
        if batch.max() > 1.0:
            batch = batch.float() / div_factor

        # Remove standardization step
        # Now apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(batch.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(batch.device)

        return (batch - mean) / std

    def forward(self, x):
        """Extract features layer by layer"""
        print("\n=== VGG Feature Extraction ===")
        print(f"Input tensor - Shape: {x.shape}")
        print(f"Range before normalization: {x.min():.3f} to {x.max():.3f}")
        
        x = self.normalize_batch(x)
        print(f"Range after normalization: {x.min():.3f} to {x.max():.3f}")
        
        # Get features with detailed debug info
        print("\nExtracting features through VGG layers:")
        h1 = self.slice1(x)
        print(f"\nSlice1 (relu1_2):")
        print(f"Shape: {h1.shape}")
        print(f"Range: {h1.min():.3f} to {h1.max():.3f}")
        print(f"Mean activation: {h1.mean():.3f}")
        
        h2 = self.slice2(h1)
        print(f"\nSlice2 (relu2_2):")
        print(f"Shape: {h2.shape}")
        print(f"Range: {h2.min():.3f} to {h2.max():.3f}")
        print(f"Mean activation: {h2.mean():.3f}")
        
        h3 = self.slice3(h2)
        print(f"\nSlice3 (relu3_3):")
        print(f"Shape: {h3.shape}")
        print(f"Range: {h3.min():.3f} to {h3.max():.3f}")
        print(f"Mean activation: {h3.mean():.3f}")
        
        h4 = self.slice4(h3)
        print(f"\nSlice4 (relu4_3):")
        print(f"Shape: {h4.shape}")
        print(f"Range: {h4.min():.3f} to {h4.max():.3f}")
        print(f"Mean activation: {h4.mean():.3f}")
        
        # Prepare outputs
        layers = [h1, h2, h3, h4]
        outputs = []
        print(f"\nReturning {min(self.layer_num, 4)} feature layers")
        
        for i in range(min(self.layer_num, 4)):
            outputs.append(layers[i])
            print(f"Layer {i+1} stats:")
            print(f"Shape: {layers[i].shape}")
            print(f"Range: {layers[i].min():.3f} to {layers[i].max():.3f}")
        
        return outputs