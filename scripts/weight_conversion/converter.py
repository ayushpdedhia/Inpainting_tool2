# scripts/weight_conversion/pconv_converter.py

import h5py
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_h5_weights(h5_path):
    """Load weights from H5 file"""
    try:
        with h5py.File(h5_path, 'r') as f:
            logger.info(f"Successfully opened {h5_path}")
            # Print structure of the h5 file
            logger.info("H5 file structure:")
            f.visit(lambda x: logger.info(x))
            return f
    except Exception as e:
        logger.error(f"Error loading H5 file: {e}")
        return None

def convert_pconv_weights(h5_path, save_path):
    """Convert PConv weights from Keras to PyTorch"""
    try:
        with h5py.File(h5_path, 'r') as f:
            state_dict = {}
            
            # Map Keras layer names to PyTorch model structure
            layer_mapping = {
                # Encoder mappings
                'p_conv2d_49': 'enc1.0',
                'p_conv2d_50': 'enc2.0',
                'p_conv2d_51': 'enc3.0',
                'p_conv2d_52': 'enc4.0',
                'p_conv2d_53': 'enc5.0',
                'p_conv2d_54': 'enc6.0',
                'p_conv2d_55': 'enc7.0',
                'p_conv2d_56': 'enc8.0',
                
                # BatchNorm mappings
                'batch_normalization_22': 'enc1.1',
                'batch_normalization_23': 'enc2.1',
                'batch_normalization_24': 'enc3.1',
                'batch_normalization_25': 'enc4.1',
                'batch_normalization_26': 'enc5.1',
                'batch_normalization_27': 'enc6.1',
                'batch_normalization_28': 'enc7.1',
                
                # Decoder mappings
                'p_conv2d_57': 'dec8.0',
                'p_conv2d_58': 'dec7.0',
                'p_conv2d_59': 'dec6.0',
                'p_conv2d_60': 'dec5.0',
                'p_conv2d_61': 'dec4.0',
                'p_conv2d_62': 'dec3.0',
                'p_conv2d_63': 'dec2.0',
                'p_conv2d_64': 'dec1.0'
            }

            # Convert each layer
            for keras_name, torch_name in layer_mapping.items():
                logger.info(f"Converting layer: {keras_name} -> {torch_name}")
                
                if keras_name.startswith('p_conv2d'):
                    # Convert convolutional layer
                    base_path = f'{keras_name}/{keras_name}'
                    
                    # Get weights and transpose them
                    weights = np.array(f[f'{base_path}/img_kernel:0'])
                    weights = np.transpose(weights, (3, 2, 0, 1))  # HWIO -> OIHW
                    
                    # Get bias
                    bias = np.array(f[f'{base_path}/bias:0'])
                    
                    state_dict[f'{torch_name}.weight'] = torch.from_numpy(weights)
                    state_dict[f'{torch_name}.bias'] = torch.from_numpy(bias)
                    
                elif keras_name.startswith('batch_normalization'):
                    # Convert batch normalization layer
                    base_path = f'{keras_name}/{keras_name}'
                    
                    # Get parameters
                    gamma = np.array(f[f'{base_path}/gamma:0'])  # scale
                    beta = np.array(f[f'{base_path}/beta:0'])    # shift
                    moving_mean = np.array(f[f'{base_path}/moving_mean:0'])
                    moving_var = np.array(f[f'{base_path}/moving_variance:0'])
                    
                    state_dict[f'{torch_name}.weight'] = torch.from_numpy(gamma)
                    state_dict[f'{torch_name}.bias'] = torch.from_numpy(beta)
                    state_dict[f'{torch_name}.running_mean'] = torch.from_numpy(moving_mean)
                    state_dict[f'{torch_name}.running_var'] = torch.from_numpy(moving_var)

            # Log the number of converted layers
            logger.info(f"Converted {len(state_dict)} parameters")
            logger.info(f"Saving weights to {save_path}")
            
            # Save the converted weights
            torch.save(state_dict, save_path)
            return True

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
def convert_vgg_weights(h5_path, save_path):
    """Convert VGG16 weights from Keras to PyTorch"""
    try:
        # Load H5 file
        h5_weights = load_h5_weights(h5_path)
        if h5_weights is None:
            return False

        # Create state dict
        state_dict = {}

        # VGG layer mapping
        vgg_mapping = {
            'block1_conv1': 'features.0',
            'block1_conv2': 'features.2',
            'block2_conv1': 'features.5',
            'block2_conv2': 'features.7',
            # Add more VGG layers as needed
        }

        # Convert weights
        for keras_name, torch_name in vgg_mapping.items():
            if keras_name in h5_weights:
                weights = np.array(h5_weights[f'{keras_name}/kernel:0'])
                bias = np.array(h5_weights[f'{keras_name}/bias:0'])
                
                weights = np.transpose(weights, (3, 2, 0, 1))
                
                state_dict[f'{torch_name}.weight'] = torch.from_numpy(weights)
                state_dict[f'{torch_name}.bias'] = torch.from_numpy(bias)
                
                logger.info(f"Converted {keras_name} -> {torch_name}")

        # Save converted weights
        torch.save(state_dict, save_path)
        logger.info(f"Successfully saved converted VGG weights to {save_path}")
        return True

    except Exception as e:
        logger.error(f"Error during VGG conversion: {e}")
        return False