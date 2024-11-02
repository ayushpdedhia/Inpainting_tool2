# src/utils/rename_test_files.py
import os
import shutil
from pathlib import Path
import json
from datetime import datetime

class TestFileOrganizer:
    def __init__(self, source_dir: str, target_dir: str):
        """
        Initialize organizer with source and target directories
        
        Args:
            source_dir: Directory containing original files
            target_dir: Directory where organized files will be stored
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.mapping_file = self.target_dir / 'file_mapping.json'
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = {
            'images': self.target_dir / 'images',
            'masks': self.target_dir / 'masks',
            'mapping': self.target_dir / 'documentation'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return dirs
        
    def create_file_mapping(self, original_files: list) -> dict:
        """Create mapping between original and new filenames"""
        mapping = {}
        
        # Counter for each file type
        counters = {
            'test_image': 1,
            'val_image': 1,
            'mask': 1
        }
        
        for file_path in original_files:
            file_path = Path(file_path)
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Determine file type (you'll need to specify criteria)
                if file_path.stem.startswith('test'):
                    new_name = f"test_image_{counters['test_image']:03d}{file_path.suffix}"
                    counters['test_image'] += 1
                elif file_path.stem.startswith('val'):
                    new_name = f"val_image_{counters['val_image']:03d}{file_path.suffix}"
                    counters['val_image'] += 1
                elif file_path.stem.startswith('mask'):
                    new_name = f"mask_{counters['mask']:03d}{file_path.suffix}"
                    counters['mask'] += 1
                else:
                    continue
                    
                mapping[str(file_path)] = new_name
                
        return mapping
        
    def save_mapping_documentation(self, mapping: dict):
        """Save file mapping documentation"""
        doc_dir = self.target_dir / 'documentation'
        doc_dir.mkdir(exist_ok=True)
        
        # Save JSON mapping
        with open(self.mapping_file, 'w') as f:
            json.dump(mapping, f, indent=4)
            
        # Create README with mapping
        readme_content = f"""# Test Files Mapping
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## File Mapping
"""
        for orig, new in mapping.items():
            readme_content += f"- {orig} -> {new}\n"
            
        with open(doc_dir / 'README.md', 'w') as f:
            f.write(readme_content)
            
    def rename_and_organize(self, mapping: dict):
        """Rename and move files according to mapping"""
        dirs = self.setup_directories()
        
        for orig_path, new_name in mapping.items():
            orig_path = Path(orig_path)
            
            # Determine target directory
            if new_name.startswith('mask'):
                target_dir = dirs['masks']
            else:
                target_dir = dirs['images']
                
            # Copy and rename file
            target_path = target_dir / new_name
            
            # Skip if source and target are the same file
            if orig_path.resolve() == target_path.resolve():
                print(f"Skipping {orig_path} as it's already in the correct location")
                continue
                
            # Copy only if target doesn't exist or is different
            if not target_path.exists() or orig_path.stat().st_size != target_path.stat().st_size:
                shutil.copy2(orig_path, target_path)
            
    def run(self):
        """Execute the complete organization process"""
        try:
            print("Starting file organization...")
            
            # Get list of files to process
            files = list(self.source_dir.glob('**/*.*'))
            
            # Create mapping
            mapping = self.create_file_mapping(files)
            
            # Save documentation
            self.save_mapping_documentation(mapping)
            
            # Rename and organize files
            self.rename_and_organize(mapping)
            
            print(f"Successfully organized {len(mapping)} files")
            print(f"Documentation saved to {self.mapping_file}")
            
        except Exception as e:
            print(f"Error during organization: {str(e)}")
            raise

def main():
    # Get source directory from user input or use default
    source_dir = input("Enter source directory (or press Enter for current directory): ").strip()
    if not source_dir:
        source_dir = "."
        
    # Get target directory
    target_dir = input("Enter target directory (or press Enter for 'data/test_samples'): ").strip()
    if not target_dir:
        target_dir = "data/test_samples"
        
    # Create and run organizer
    organizer = TestFileOrganizer(source_dir, target_dir)
    organizer.run()

if __name__ == "__main__":
    main()