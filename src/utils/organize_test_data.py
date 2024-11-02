# src/utils/organize_test_data.py
import os
import shutil
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_test_directories():
    """Create necessary directories for test data"""
    base_dir = Path("data/test_samples")
    dirs = {
        'images': base_dir / 'images',
        'masks': base_dir / 'masks',
        'test_outputs': base_dir / 'test_outputs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
        
    return dirs

def verify_test_files(dirs):
    """Verify that all required test files exist"""
    required_files = {
        'images': [
            'test_image_001.JPEG', 'test_image_002.JPEG', 'test_image_003.JPEG',
            'test_image_004.JPEG', 'test_image_005.JPEG',
            'val_image_001.JPEG', 'val_image_002.JPEG', 'val_image_003.JPEG',
            'val_image_004.JPEG', 'val_image_005.JPEG'
        ],
        'masks': [
            'mask_large.png', 'mask_edge.png', 'mask_thick.png',
            'mask_thin.png', 'mask_corner.png', 'mask_small.png'  # Fixed spelling to match actual file
        ]
    }
    
    all_files_present = True
    for dir_type, files in required_files.items():
        dir_path = dirs[dir_type]
        for file in files:
            if not (dir_path / file).exists():
                logger.warning(f"Missing required file: {dir_path / file}")
                all_files_present = False
    
    if all_files_present:
        logger.info("All required test files are present")
    return all_files_present

def organize_test_files():
    """Organize test files into appropriate directories"""
    try:
        # Setup directories
        dirs = setup_test_directories()
        
        # Define file mappings
        mask_files = {
            'mask_large.png': 'Large contiguous region mask',
            'mask_edge.png': 'Edge pattern mask',
            'mask_thick.png': 'Thick pattern mask',
            'mask_thin.png': 'Thin scattered mask',
            'mask_corner.png': 'Corner pattern mask',
            'mask_small.png': 'Small scattered mask'  # Fixed spelling to match actual file
        }
        
        test_files = {
            'test_image_001.JPEG': 'Microscope grayscale image',
            'test_image_002.JPEG': 'Purple flower with droplets',
            'test_image_003.JPEG': 'Sea cucumber texture',
            'test_image_004.JPEG': 'Magpie bird on fence',
            'test_image_005.JPEG': 'Vintage TV purple wall'
        }
        
        val_files = {
            'val_image_001.JPEG': 'Sheltie dog portrait',
            'val_image_002.JPEG': 'Decorative bowl with soup',
            'val_image_003.JPEG': 'Snake on road',
            'val_image_004.JPEG': 'Flying eagle',
            'val_image_005.JPEG': 'Military vehicle'
        }
        
        def create_readme(dir_path, content):
            """Create README file with content"""
            readme_path = dir_path / 'README.md'
            with open(readme_path, 'w') as f:
                f.write(content)
            logger.info(f"Created README: {readme_path}")
        
        # Create README files
        create_readme(
            dirs['masks'],
            "# Test Masks\n\n" + "\n".join([f"- {k}: {v}" for k, v in mask_files.items()])
        )
        
        create_readme(
            dirs['images'],
            "# Test Images\n\n" +
            "## Test Set\n" +
            "\n".join([f"- {k}: {v}" for k, v in test_files.items()]) +
            "\n\n## Validation Set\n" +
            "\n".join([f"- {k}: {v}" for k, v in val_files.items()])
        )
        
        # Create description file
        description_path = dirs['images'].parent / 'test_samples_info.txt'
        with open(description_path, 'w') as f:
            f.write("MASKS:\n\n")
            for name, desc in mask_files.items():
                f.write(f"{name}: {desc}\n")
            
            f.write("\nTEST IMAGES:\n\n")
            for name, desc in test_files.items():
                f.write(f"{name}: {desc}\n")
            
            f.write("\nVALIDATION IMAGES:\n\n")
            for name, desc in val_files.items():
                f.write(f"{name}: {desc}\n")
        
        logger.info(f"Created test samples info: {description_path}")
        
        return dirs
        
    except Exception as e:
        logger.error(f"Error organizing test files: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        dirs = organize_test_files()
        logger.info("Successfully organized test directories:")
        for name, path in dirs.items():
            logger.info(f"- {name}: {path}")
    except Exception as e:
        logger.error(f"Error organizing files: {str(e)}")