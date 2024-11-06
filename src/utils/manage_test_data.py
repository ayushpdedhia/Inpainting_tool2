# src/utils/manage_test_data.py
# python -m src.utils.manage_test_data       
import logging
from pathlib import Path
from .organize_test_data import organize_test_files, verify_test_files
from .rename_test_files import TestFileOrganizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_test_environment():
    """Setup test environment without model dependencies"""
    try:
        logger.info("Setting up test environment...")
        
        # Setup directories and verify files
        dirs = organize_test_files()
        logger.info("Base directories created successfully")
        
        # Check if files already exist and are properly organized
        if verify_test_files(dirs):
            logger.info("Test files already organized, skipping reorganization")
            return True
            
        # Only reorganize if necessary
        organizer = TestFileOrganizer(
            source_dir=dirs['images'].parent,
            target_dir=dirs['images'].parent
        )
        organizer.run()
        
        logger.info("Test environment setup complete")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up test environment: {e}")
        return False

if __name__ == "__main__":
    success = setup_test_environment()
    if success:
        print("Test environment setup completed successfully")
    else:
        print("Test environment setup failed")