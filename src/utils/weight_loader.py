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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def _load_config(config_path):
        """Load configuration from yaml file"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_model_weights(self, model):
        """
        Load pretrained weights for PConv model
        
        Args:
            model: PConvUNet model instance
        Returns:
            model: Model with loaded weights
        """
        try:
            # Get weight paths from config
            unet_weights_path = self.config['paths']['unet_weights']
            vgg_weights_path = self.config['paths']['vgg_weights']

            # Load UNet weights
            if os.path.exists(unet_weights_path):
                checkpoint = torch.load(unet_weights_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, OrderedDict):
                    state_dict = checkpoint
                else:
                    state_dict = checkpoint.get('state_dict', checkpoint)

                # Clean up state dict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')  # Remove DataParallel wrapper
                    new_state_dict[name] = v

                # Load weights
                model.load_state_dict(new_state_dict, strict=False)
                logger.info(f"Successfully loaded UNet weights from {unet_weights_path}")

            # Move model to device
            model = model.to(self.device)
            return model

        except Exception as e:
            logger.error(f"Error loading weights: {str(e)}")
            raise

    def get_device(self):
        """Return current device"""
        return self.device