import torch
import torch.nn.functional as F
import numpy as np
import os
import traceback
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Any, List
from torch.utils.data import DataLoader

from ..models.pconv.loss import PConvLoss
from ..models.pconv.models.pconv_unet import PConvUNet
from ..utils.weight_loader import WeightLoader
from ..utils.data_loader import InpaintingDataset, get_data_loader

class ModelManager:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ModelManager
        
        Args:
            config_path: Optional path to config file. If None, uses default path.
        """
        self.models: Dict[str, torch.nn.Module] = {}
        self.available_models = {
            'partial convolutions': 'pdvgg16_bn'
        }
        
        # Get config path
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, '..', '..', 'config.yaml')
        
        # Initialize weight loader and device
        try:
            self.weight_loader = WeightLoader(config_path)
            # Use device from config if available
            self.device = torch.device(self.weight_loader.config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        except Exception as e:
            print(f"Error initializing WeightLoader: {str(e)}")
            self.weight_loader = None
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

            weights_loaded = False

            # First try using WeightLoader
            if self.weight_loader is not None:
                try:
                    model = self.weight_loader.load_model_weights(model, load_vgg=True)
                    print("Successfully loaded model weights using WeightLoader")
                    weights_loaded = True
                except Exception as e:
                    print(f"WeightLoader failed, falling back to direct loading: {str(e)}")
            
            # Fallback to direct loading if WeightLoader failed
            if not weights_loaded:
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

                        # Setup VGG weights path for future use
                        self.vgg_weights_path = os.path.join(base_weights_dir, 'vgg16', 'vgg16_weights.pth')
                    except Exception as e:
                        print(f"Error loading model weights: {str(e)}")
                        raise
                else:
                    raise FileNotFoundError(f"Model weights not found at {weights_path}")
            
            model.eval()
            self.models['partial convolutions'] = model
            
        except Exception as e:
            print(f"Error in model initialization: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise

        
    def create_data_loader(self,
                          image_dir: str,
                          mask_dir: Optional[str] = None,
                          batch_size: int = 1,
                          image_size: Tuple[int, int] = (512, 512),
                          num_workers: int = 4,
                          shuffle: bool = False) -> DataLoader:
        """Create a data loader for processing multiple images"""
        return get_data_loader(
            image_dir=image_dir,
            mask_dir=mask_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            shuffle=shuffle
        )
    
    def process_batch(self, 
                     data_loader: DataLoader, 
                     model_name: str = 'partial convolutions') -> List[np.ndarray]:
        """Process a batch of images using the model"""
        results = []
        for batch in data_loader:
            image = batch['image'].numpy()
            mask = batch['mask'].numpy()
            
            for i in range(len(image)):
                result = self.inpaint(image[i], mask[i], model_name)
                results.append(result)
                
        return results
    
    def preprocess_inputs(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess numpy inputs to torch tensors.
        
        Args:
            image (np.ndarray): Input image in numpy format with shape [H, W, C], 
                            where C=3 for RGB images. Values should be in range [0, 1].
            mask (np.ndarray): Binary mask with shape [H, W] or [H, W, 1], 
                            where 1 indicates regions to keep and 0 indicates regions to inpaint.
                            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Preprocessed image and mask tensors with shapes 
            [1, C, H, W] and [1, 1, H, W] respectively.
            
        Raises:
            ValueError: If inputs don't match expected formats, shapes, or value ranges.
            TypeError: If inputs aren't numpy arrays.
        """
        try:
            # Type validation
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Expected image to be numpy array, got {type(image)}")
            if not isinstance(mask, np.ndarray):
                raise TypeError(f"Expected mask to be numpy array, got {type(mask)}")

            # Print shapes for debugging
            print(f"Input image shape: {image.shape}")
            print(f"Input mask shape: {mask.shape}")

            # Shape validation
            if len(image.shape) != 3:
                raise ValueError(f"Expected image with shape [H, W, C], got {image.shape}")
            if len(mask.shape) not in [2, 3]:
                raise ValueError(f"Expected mask with shape [H, W] or [H, W, 1], got {mask.shape}")
            
            # Channel validation
            if image.shape[2] != 3:
                raise ValueError(f"Expected 3 channels (RGB) for image, got {image.shape[2]}")

            # Dimension matching
            img_h, img_w = image.shape[:2]
            mask_h, mask_w = mask.shape[:2]
            if img_h != mask_h or img_w != mask_w:
                raise ValueError(
                    f"Image dimensions ({img_h}, {img_w}) must match mask dimensions ({mask_h}, {mask_w})"
                )

            # Value range validation
            if np.min(image) < 0 or np.max(image) > 1:
                raise ValueError(f"Image values must be in range [0, 1], got range [{np.min(image)}, {np.max(image)}]")
            if np.min(mask) < 0 or np.max(mask) > 1:
                raise ValueError(f"Mask values must be in range [0, 1], got range [{np.min(mask)}, {np.max(mask)}]")

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

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise

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
        Inpaint the image using the selected model.
        
        Args:
            image (np.ndarray): Input image with shape [H, W, C] in range [0, 1]
            mask (np.ndarray): Binary mask with shape [H, W] where 1 indicates regions to keep
            model_name (str, optional): Name of the model to use. Defaults to 'partial convolutions'.
            
        Returns:
            np.ndarray: Inpainted image with shape [H, W, C] in range [0, 1]
            
        Raises:
            ValueError: For invalid inputs or model configuration
            TypeError: For incorrect input types
            RuntimeError: For model execution errors
        """
        try:
            # Input validation
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Expected image to be numpy array, got {type(image)}")
            if not isinstance(mask, np.ndarray):
                raise TypeError(f"Expected mask to be numpy array, got {type(mask)}")
            if not isinstance(model_name, str):
                raise TypeError(f"Expected model_name to be string, got {type(model_name)}")

            # Validate model availability
            model_name = model_name.lower()
            if model_name not in self.models:
                raise ValueError(
                    f"Model {model_name} not available. Available models: {list(self.models.keys())}"
                )

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
                    raise TypeError(f"Model output is not a tensor. Got {type(output)}")
                
                if len(output.shape) != 4:
                    raise ValueError(f"Expected output shape [N, C, H, W], got {output.shape}")
                
                if output.shape[1] != 3:
                    raise ValueError(f"Expected 3 output channels, got {output.shape[1]}")
                
                # Calculate loss for monitoring
                loss, loss_dict = criterion(output, image_tensor, mask_tensor)
                print("Loss components:", {k: v.item() for k, v in loss_dict.items()})
                
                # Validate output values
                if torch.isnan(output).any():
                    raise ValueError("Model output contains NaN values")
                
                if torch.isinf(output).any():
                    raise ValueError("Model output contains infinite values")
                
                # Postprocess output
                result = self.postprocess_output(output, image_tensor, mask_tensor)
                
                # Final validation of result
                if not isinstance(result, np.ndarray):
                    raise TypeError(f"Expected numpy array output, got {type(result)}")
                
                if result.shape != image.shape:
                    raise ValueError(f"Output shape {result.shape} doesn't match input shape {image.shape}")
            
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