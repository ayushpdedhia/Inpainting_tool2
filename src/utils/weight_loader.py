import torch
import os
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class WeightLoader:
    def __init__(self, config_path):
        """
        Initialize weight loader with configuration
        
        Args:
            config_path: Path to config.yaml
        """
        self.config = self._load_config(config_path)
        # Modify device initialization to include index 0 for CUDA
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def _load_config(config_path):
        """Load configuration from yaml file"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def _log_checkpoint_info(self, checkpoint, path):
        """Log useful information from checkpoint"""
        if 'epoch' in checkpoint:
            logger.info(f"Checkpoint from epoch: {checkpoint['epoch']}")
        if 'best_prec1' in checkpoint:
            logger.info(f"Best precision: {checkpoint['best_prec1']:.2f}")
        if 'date' in checkpoint:
            logger.info(f"Checkpoint date: {checkpoint['date']}")
        logger.info(f"Checkpoint size: {os.path.getsize(path) / 1024:.2f} KB")

    def load_model_weights(self, model, load_vgg=True):
        """
        Load pretrained weights for PConv model
        
        Args:
            model: PConvUNet model instance
            load_vgg: Whether to load VGG weights for loss computation
        Returns:
            model: Model with loaded weights
        """
        try:
            # Get weight paths from config
            unet_weights_path = self.config['paths']['unet_weights']
            vgg_weights_path = self.config['paths']['vgg_weights']

            # First move model to device
            model = model.to(self.device)

            # Load UNet weights
            if os.path.exists(unet_weights_path):
                checkpoint = torch.load(unet_weights_path, map_location=self.device, weights_only=True)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, OrderedDict):
                    state_dict = checkpoint
                else:
                    state_dict = checkpoint.get('state_dict', checkpoint)

                # Clean up state dict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')  # Remove DataParallel wrapper
                    # Ensure tensor is on correct device
                    new_state_dict[name] = v.to(self.device)

                # Load weights
                model.load_state_dict(new_state_dict, strict=False)
                logger.info(f"Successfully loaded UNet weights from {unet_weights_path}")

            # Load VGG weights if needed
            if load_vgg and hasattr(model, 'vgg') and os.path.exists(vgg_weights_path):
                vgg_checkpoint = torch.load(vgg_weights_path, map_location=self.device, weights_only=True)
                
                if isinstance(vgg_checkpoint, OrderedDict):
                    vgg_state_dict = vgg_checkpoint
                else:
                    vgg_state_dict = vgg_checkpoint.get('state_dict', vgg_checkpoint)
                
                # Clean up VGG state dict
                vgg_new_state_dict = OrderedDict()
                for k, v in vgg_state_dict.items():
                    name = k.replace('module.', '')
                    # Ensure tensor is on correct device
                    vgg_new_state_dict[name] = v.to(self.device)
                
                # Load VGG weights
                model.vgg.load_state_dict(vgg_new_state_dict, strict=False)
                logger.info(f"Successfully loaded VGG weights from {vgg_weights_path}")

            return model
        except Exception as e:
            logger.error(f"Error loading weights: {str(e)}")
            raise

    def get_device(self):
        """Return current device"""
        return self.device