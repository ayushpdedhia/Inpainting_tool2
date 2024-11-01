import torch
import torch.nn.functional as F
import numpy as np
import os
import traceback
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Any

from ..models.pconv.loss import PConvLoss
from ..models.pconv.models.pconv_unet import PConvUNet
from ..utils.weight_loader import WeightLoader

class ModelManager:
    def __init__(self):
        self.models: Dict[str, torch.nn.Module] = {}
        self.available_models = {
            'partial convolutions': 'pdvgg16_bn'
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.vgg_weights_path = None
        self.load_models()
    
    def load_models(self) -> None:
        """Initialize and load all available models"""
        try:
            # Get absolute path of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_weights_dir = os.path.join(current_dir, '..', '..', 'weights', 'pconv')
            
            print(f"Base weights directory: {base_weights_dir}")
            if not os.path.exists(base_weights_dir):
                print(f"Directory structure:")
                print(f"Current directory: {current_dir}")
                print("Available directories:")
                print(os.listdir(current_dir))
                raise FileNotFoundError(f"Base weights directory not found at {base_weights_dir}")
            
            # Setup VGG weights path
            self.vgg_weights_path = os.path.join(base_weights_dir, 'vgg16', 'vgg16_weights.pth')
            
            # Load PConv model
            model = PConvUNet()
            model = model.to(self.device)
            
            # Load model weights
            weights_path = os.path.join(base_weights_dir, 'unet', 'model_weights.pth')
            if os.path.exists(weights_path):
                try:
                    checkpoint = torch.load(weights_path, map_location=self.device)
                    if isinstance(checkpoint, OrderedDict):
                        state_dict = checkpoint
                    else:
                        state_dict = checkpoint.get('state_dict', checkpoint)
                    
                    # Create a new state dict with the correct keys
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        # Remove 'module.' prefix if present
                        name = k.replace('module.', '')
                        if name.startswith('features.'):
                            # Map features to encoder
                            new_name = name.replace('features.', 'encoder.features.')
                            new_state_dict[new_name] = v
                        else:
                            new_state_dict[name] = v
                    
                    model.load_state_dict(new_state_dict, strict=False)
                    print(f"Successfully loaded model weights from {weights_path}")
                except Exception as e:
                    print(f"Error loading model weights: {str(e)}")
                    raise
            
            model.eval()
            self.models['partial convolutions'] = model
            
        except Exception as e:
            print(f"Error in model initialization: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise

    def preprocess_inputs(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess numpy inputs to torch tensors"""
        # Print shapes for debugging
        print(f"Input image shape: {image.shape}")
        print(f"Input mask shape: {mask.shape}")
        
        # Validate input shapes
        if len(image.shape) != 3:
            raise ValueError(f"Expected image with shape [H, W, C], got {image.shape}")
        if len(mask.shape) not in [2, 3]:
            raise ValueError(f"Expected mask with shape [H, W] or [H, W, 1], got {mask.shape}")

        # Convert to float32
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        # Ensure mask is 2D
        if len(mask.shape) == 3:
            mask = mask.squeeze(-1)

        # Invert mask for processing (0 for regions to keep, 1 for regions to inpaint)
        inv_mask = 1 - mask

        # Convert to tensors and reshape
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        mask_tensor = torch.from_numpy(inv_mask).unsqueeze(0).unsqueeze(0)

        # Print tensor shapes for debugging
        print(f"Preprocessed image tensor shape: {image_tensor.shape}")
        print(f"Preprocessed mask tensor shape: {mask_tensor.shape}")

        return image_tensor, mask_tensor

    def postprocess_output(self, output: torch.Tensor, original: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        """Postprocess model output"""
        # Handle size mismatch if necessary
        if output.shape[-2:] != original.shape[-2:]:
            print(f"Resizing output from {output.shape[-2:]} to {original.shape[-2:]}")
            output = F.interpolate(output, size=original.shape[-2:],
                                mode='bilinear', align_corners=False)
        
        # Create composite output
        comp = output * (1 - mask) + original * mask
        
        # Convert to numpy
        result = comp.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Ensure output is in valid range
        return np.clip(result, 0, 1)

    def inpaint(self, image: np.ndarray, mask: np.ndarray, model_name: str = 'partial convolutions') -> np.ndarray:
        """
        Inpaint the image using the selected model
        Args:
            image: Input image [H, W, C]
            mask: Binary mask [H, W] where 1 indicates regions to keep
            model_name: Name of the model to use
        Returns:
            Inpainted image [H, W, C]
        """
        try:
            # Validate model availability
            model_name = model_name.lower()
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not available. Available models: {list(self.models.keys())}")

            # Preprocess inputs
            image_tensor, mask_tensor = self.preprocess_inputs(image, mask)
            
            # Move to device
            image_tensor = image_tensor.to(self.device)
            mask_tensor = mask_tensor.to(self.device)
            
            # Get model and initialize loss
            model = self.models[model_name]
            criterion = PConvLoss(device=self.device)
            
            with torch.no_grad():
                # Forward pass
                output = model(image_tensor, mask_tensor)
                
                # Debug output shape
                print(f"Raw output shape: {output.shape}")
                
                # Validate output
                if not isinstance(output, torch.Tensor):
                    raise ValueError(f"Model output is not a tensor. Got {type(output)}")
                
                if len(output.shape) != 4:
                    raise ValueError(f"Expected output shape [N, C, H, W], got {output.shape}")
                
                # Calculate loss for monitoring
                loss, loss_dict = criterion(output, image_tensor, mask_tensor)
                print("Loss components:", {k: v.item() for k, v in loss_dict.items()})
                
                # Postprocess output
                result = self.postprocess_output(output, image_tensor, mask_tensor)
            
            return result
            
        except Exception as e:
            print(f"Error during inpainting: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise

    def get_available_models(self) -> Dict[str, str]:
        """Return dictionary of available models"""
        return self.available_models.copy()

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        return {
            'name': model_name,
            'type': model.__class__.__name__,
            'parameters': sum(p.numel() for p in model.parameters()),
            'device': next(model.parameters()).device
        }