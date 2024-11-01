import os
import shutil
from pathlib import Path

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
        
    return dirs

def organize_test_files():
    """Organize test files into appropriate directories"""
    dirs = setup_test_directories()
    
    # Define file mappings
    mask_files = {
        'mask_large.png': 'Large contiguous region mask',
        'mask_edge.png': 'Edge pattern mask',
        'mask_thick.png': 'Thick pattern mask',
        'mask_thin.png': 'Thin scattered mask',
        'mask_corner.png': 'Corner pattern mask',
        'mask_small.png': 'Small scattered mask'
    }
    
    test_files = {
        'test_image_001.jpg': 'Microscope grayscale image',
        'test_image_002.jpg': 'Purple flower with droplets',
        'test_image_003.jpg': 'Sea cucumber texture',
        'test_image_004.jpg': 'Magpie bird on fence',
        'test_image_005.jpg': 'Vintage TV purple wall'
    }
    
    val_files = {
        'val_image_001.jpg': 'Sheltie dog portrait',
        'val_image_002.jpg': 'Decorative bowl with soup',
        'val_image_003.jpg': 'Snake on road',
        'val_image_004.jpg': 'Flying eagle',
        'val_image_005.jpg': 'Military vehicle'
    }
    
    def create_readme(dir_path, content):
        with open(dir_path / 'README.md', 'w') as f:
            f.write(content)
    
    # Create README files
    create_readme(dirs['masks'], "# Test Masks\n\n" + 
                 "\n".join([f"- {k}: {v}" for k, v in mask_files.items()]))
    
    create_readme(dirs['images'], "# Test Images\n\n" +
                 "## Test Set\n" +
                 "\n".join([f"- {k}: {v}" for k, v in test_files.items()]) +
                 "\n\n## Validation Set\n" +
                 "\n".join([f"- {k}: {v}" for k, v in val_files.items()]))
    
    return dirs

if __name__ == "__main__":
    try:
        dirs = organize_test_files()
        print("Successfully organized test directories:")
        for name, path in dirs.items():
            print(f"- {name}: {path}")
    except Exception as e:
        print(f"Error organizing files: {str(e)}")