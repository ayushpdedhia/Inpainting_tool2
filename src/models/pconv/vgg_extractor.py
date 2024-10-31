import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models  # Added this import

def gram_matrix(input_tensor):
    """
    Compute Gram matrix efficiently using baddbmm
    
    Args:
        input_tensor: Input tensor of shape (batch_size, channels, height, width)
    Returns:
        Gram matrix
    """
    (b, ch, h, w) = input_tensor.size()
    features = input_tensor.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    
    # More efficient computation using baddbmm
    gram = torch.zeros(b, ch, ch).type(features.type())
    gram = torch.baddbmm(
        gram, features, features_t,
        beta=0,
        alpha=1. / (ch * h * w)
    )
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
    def normalize_batch(batch, div_factor=1.0):
        """Normalize batch using ImageNet stats"""
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        batch = batch / div_factor
        batch = (batch - mean) / std
        return batch

    def forward(self, x):
        """Extract features layer by layer"""
        x = self.normalize_batch(x)
        h = self.slice1(x)
        h1 = h
        h = self.slice2(h)
        h2 = h
        h = self.slice3(h)
        h3 = h
        h = self.slice4(h)
        h4 = h
        outputs = []
        if self.layer_num >= 1: outputs.append(h1)
        if self.layer_num >= 2: outputs.append(h2)
        if self.layer_num >= 3: outputs.append(h3)
        if self.layer_num >= 4: outputs.append(h4)
        return outputs