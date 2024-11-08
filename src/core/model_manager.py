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
        self.weight_loader = None  # Initialize to None first
        self.vgg_weights_path = None
        
        # Get config path
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, '..', '..', 'config.yaml')
        
        # Initialize weight loader and device
        try:
            self.weight_loader = WeightLoader(config_path)
            # Use device from config if available, with explicit CUDA device index
            if self.weight_loader.config['model'].get('device', '').startswith('cuda'):
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        except Exception as e:
            print(f"Error initializing WeightLoader: {str(e)}")
            # Default to CUDA:0 if available
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Load models after everything is initialized
        self.load_models()
        self.check_gpu_status()

    def check_gpu_status(self):
        print(f"Current GPU Device: {torch.cuda.current_device()}")
        print(f"GPU Device Name: {torch.cuda.get_device_name()}")
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
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
                        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
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
        
        Mask Convention:
        Input from canvas/processor: white (255) = inpaint, black (0) = keep
        Output tensor: 0 = inpaint, 1 = keep original
        """
        try:
            # Debug input state
            print("\n=== Preprocessing Input Stats ===")
            print(f"Image - Shape: {image.shape}, Range: [{np.min(image)}, {np.max(image)}]")
            print(f"Mask - Shape: {mask.shape}, Range: [{np.min(mask)}, {np.max(mask)}]")
            print(f"Mask unique values: {np.unique(mask)}")

            # Type validation
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Expected image to be numpy array, got {type(image)}")
            if not isinstance(mask, np.ndarray):
                raise TypeError(f"Expected mask to be numpy array, got {type(mask)}")

            # Print shapes for debugging
            print(f"Input image shape: {image.shape}")
            print(f"Input mask shape: {mask.shape}")

            # Handle image format
            if len(image.shape) == 4:  # BCHW format
                if image.shape[1] != 3:
                    raise ValueError(f"Expected 3 channels (RGB), got {image.shape[1]}")
                image = image[0] if image.shape[0] > 1 else image
                image_tensor = torch.from_numpy(image)
            else:  # HWC format
                if len(image.shape) != 3 or image.shape[2] != 3:
                    raise ValueError(f"Expected shape [H,W,3], got {image.shape}")
                if np.min(image) < 0 or np.max(image) > 1:
                    raise ValueError(f"Image values must be in [0,1], got [{np.min(image)}, {np.max(image)}]")
                image = image.astype(np.float32)
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

            # Handle mask format
            if len(mask.shape) == 4:  # BCHW format
                mask = mask[0] if mask.shape[0] > 1 else mask
                mask = mask[:1] if mask.shape[1] != 1 else mask
                mask_tensor = torch.from_numpy(mask)
            else:
                # Convert to 2D if needed
                mask = mask.squeeze() if len(mask.shape) == 3 else mask
                if len(mask.shape) != 2:
                    raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
                
                # Ensure matching dimensions
                if mask.shape != (image_tensor.shape[2], image_tensor.shape[3]):
                    raise ValueError(f"Mask shape {mask.shape} doesn't match image {(image_tensor.shape[2], image_tensor.shape[3])}")
                
                # Convert to binary mask: white (255) → 0 (inpaint), black (0) → 1 (keep)
                if mask.max() > 1:
                    mask = mask.astype(np.float32) / 255.0
                mask = (1.0 - mask).astype(np.float32)  # Convert: 0=keep to 1=keep
                # mask = (mask < 0.5).astype(np.float32)  # Invert convention
                mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

            # Convert to float32
            image_tensor = image_tensor.float()
            mask_tensor = mask_tensor.float()
    

            # Print tensor shapes for debugging
            print("\n=== Preprocessed Tensor Stats ===")
            print(f"Image tensor - Shape: {image_tensor.shape}, Range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
            print(f"Mask tensor - Shape: {mask_tensor.shape}, Range: [{mask_tensor.min():.3f}, {mask_tensor.max():.3f}]")
            print(f"Mask tensor unique values: {torch.unique(mask_tensor).cpu().numpy()}")

            return image_tensor, mask_tensor

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise

    def postprocess_output(self, output: torch.Tensor, original: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        """
        Postprocess model output.
        
        Mask convention:
        1 = keep original pixels
        0 = use inpainted pixels
        """
        with torch.no_grad():
            # Debug before compositing
            print("\n=== Postprocessing Debug ===")
            print(f"Output tensor - Range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"Original tensor - Range: [{original.min():.3f}, {original.max():.3f}]")
            print(f"Mask tensor - Range: [{mask.min():.3f}, {mask.max():.3f}]")
            print(f"Mask unique values: {torch.unique(mask).cpu().numpy()}")
            
            # Handle size mismatch
            if output.shape[-2:] != original.shape[-2:]:
                output = F.interpolate(output, size=original.shape[-2:],
                                    mode='bilinear', align_corners=False)

            # Compose output: mask=1 keeps original, mask=0 uses inpainted
            comp = original * mask + output * (1 - mask)
            # Debug after compositing
            print(f"Composition - Range: [{comp.min():.3f}, {comp.max():.3f}]")
            print(f"Original regions - Range: [{(comp * mask).min():.3f}, {(comp * mask).max():.3f}]")
            print(f"Inpainted regions - Range: [{(comp * (1-mask)).min():.3f}, {(comp * (1-mask)).max():.3f}]")
            
            # Convert to numpy and transpose to HWC
            result = comp.squeeze(0).cpu().numpy()
            result = np.transpose(result, (1, 2, 0)) if result.shape[0] == 3 else result
            
            # Debug final output
            print(f"Final output - Shape: {result.shape}, Range: [{result.min():.3f}, {result.max():.3f}]")
                
            return result

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
        self.check_gpu_status()
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
                # Debug masks
                print("\n=== Mask Statistics ===")
                print(f"Input mask unique values: {np.unique(mask)}")
                print(f"Processed mask tensor unique values: {torch.unique(mask_tensor).cpu().numpy()}")
            
                # Forward pass
                output = model(image_tensor, mask_tensor)

                # Debug output
                print("\n=== Output Statistics ===")
                print(f"Output min/max values: {output.min().item():.6f}, {output.max().item():.6f}")
                print(f"Output mean value: {output.mean().item():.6f}")
        
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
                
                # Get input shape format (BCHW or HWC)
                input_is_bchw = len(image.shape) == 4
                
                # Postprocess output
                result = self.postprocess_output(output, image_tensor, mask_tensor)

                # Validate output shape based on input format
                if input_is_bchw:
                    expected_shape = (image.shape[2], image.shape[3], image.shape[1])  # Convert BCHW to HWC
                else:
                    expected_shape = image.shape  # Keep HWC as is
                
                # Final validation of result
                if not isinstance(result, np.ndarray):
                    raise TypeError(f"Expected numpy array output, got {type(result)}")
                
                if result.shape != expected_shape:
                    raise ValueError(
                        f"Output shape {result.shape} doesn't match expected shape {expected_shape}"
                    )
                
                # Add comparison
                self.compare_images(image, result, mask)
            
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
    
    def compare_images(self, input_image: np.ndarray, output_image: np.ndarray, mask: np.ndarray) -> None:
        """
        Compare input and output images.
        
        Mask convention:
        1 = valid/original regions
        0 = inpainted regions
        """
        try:
            # Convert input to HWC if it's in BCHW format
            if len(input_image.shape) == 4:
                input_image = input_image[0].transpose(1, 2, 0)
            elif len(input_image.shape) == 3 and input_image.shape[0] == 3:
                input_image = input_image.transpose(1, 2, 0)
                
            print("\n=== Image Comparison ===")
            print(f"Input shape: {input_image.shape}")
            print(f"Output shape: {output_image.shape}")
            
            if input_image.shape != output_image.shape:
                print(f"Shape mismatch: Input {input_image.shape} vs Output {output_image.shape}")
                return

            # Calculate differences
            diff = np.abs(input_image - output_image)
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            
            # Ensure 2D mask
            mask = mask.squeeze()
            mask = (mask < 0.5).astype(np.float32)  # Convert to binary: 0=inpaint, 1=keep
            
            # Calculate differences in inpainted regions (mask=0)
            inpaint_diff = diff * (1 - mask)[..., None]
            inpaint_mean = np.mean(inpaint_diff[mask < 0.5]) if np.any(mask < 0.5) else 0
            inpaint_max = np.max(inpaint_diff) if np.any(mask < 0.5) else 0
            
            # Print statistics
            print("\n=== Difference Statistics ===")
            print(f"Overall - Mean: {mean_diff:.6f}, Max: {max_diff:.6f}")
            print(f"Inpainted regions - Mean: {inpaint_mean:.6f}, Max: {inpaint_max:.6f}")
            
            # Check for potential issues
            if inpaint_mean < 0.001:
                print("\nWARNING: Very small differences in inpainted regions")
                print("This might indicate improper inpainting")
                
                # Sample some inpainted pixels
                if np.any(mask < 0.5):
                    coords = np.where(mask < 0.5)
                    samples = np.random.choice(len(coords[0]), min(5, len(coords[0])), replace=False)
                        
                    print("\nSample pixel comparisons:")
                    for idx in samples:
                        y, x = coords[0][idx], coords[1][idx]
                        print(f"Position ({x}, {y}):")
                        print(f"  Input: {input_image[y, x]}")
                        print(f"  Output: {output_image[y, x]}")
                            
        except Exception as e:
            print(f"Error in image comparison: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")