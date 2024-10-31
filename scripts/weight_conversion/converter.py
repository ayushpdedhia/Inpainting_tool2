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
            logger.info("H5 file structure:")
            f.visit(lambda x: logger.info(x))
            return f
    except Exception as e:
        logger.error(f"Error loading H5 file: {e}")
        return None

def convert_pconv_weights(h5_path, save_path):
    """Convert PConv weights from Keras to PyTorch. Handles missing BatchNorm layers."""
    try:
        with h5py.File(h5_path, 'r') as f:
            state_dict = {}
            
            # Only map layers that exist in the H5 file
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
                
                # Encoder BatchNorm mappings
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
                'p_conv2d_64': 'dec1.0',
                
                # Note: Decoder BatchNorm layers will be initialized with default values
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

            # Initialize missing BatchNorm layers with default values
            missing_bn_layers = {
                'dec4.1': 256,
                'dec3.1': 128,
                'dec2.1': 64,
                'dec1.1': 3
            }

            for layer_name, num_features in missing_bn_layers.items():
                if f'{layer_name}.weight' not in state_dict:
                    state_dict[f'{layer_name}.weight'] = torch.ones(num_features)
                    state_dict[f'{layer_name}.bias'] = torch.zeros(num_features)
                    state_dict[f'{layer_name}.running_mean'] = torch.zeros(num_features)
                    state_dict[f'{layer_name}.running_var'] = torch.ones(num_features)

            logger.info(f"Converted {len(state_dict)} parameters")
            logger.info(f"Saving weights to {save_path}")
            
            torch.save(state_dict, save_path)
            return True

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def convert_vgg_weights(h5_path, save_path):
    """Convert VGG16 weights from Keras to PyTorch. Maps features correctly."""
    try:
        with h5py.File(h5_path, 'r') as f:
            state_dict = {}

            # Map VGG layers, considering the module structure
            vgg_mapping = {
                'block1_conv1': 'features.module.0',
                'block1_conv2': 'features.module.3',
                'block2_conv1': 'features.module.7',
                'block2_conv2': 'features.module.10',
                'block3_conv1': 'features.10',
                'block3_conv2': 'features.12',
                'block3_conv3': 'features.14',
                'block4_conv1': 'features.17',
                'block4_conv2': 'features.19',
                'block4_conv3': 'features.21',
                'block5_conv1': 'features.24',
                'block5_conv2': 'features.26',
                'block5_conv3': 'features.28'
            }

            for keras_name, torch_name in vgg_mapping.items():
                if keras_name in f:
                    # Convert weights
                    weights = np.array(f[f'{keras_name}/{keras_name}/kernel:0'])
                    bias = np.array(f[f'{keras_name}/{keras_name}/bias:0'])
                    
                    # Transpose weights from Keras format to PyTorch format
                    weights = np.transpose(weights, (3, 2, 0, 1))
                    
                    state_dict[f'{torch_name}.weight'] = torch.from_numpy(weights)
                    state_dict[f'{torch_name}.bias'] = torch.from_numpy(bias)
                    
                    logger.info(f"Converted {keras_name} -> {torch_name}")
                else:
                    logger.warning(f"Layer {keras_name} not found in H5 file")

            torch.save(state_dict, save_path)
            logger.info(f"Successfully saved converted VGG weights to {save_path}")
            return True

    except Exception as e:
        logger.error(f"Error during VGG conversion: {e}")
        return False