# src/utils/manage_test_data.py
from .organize_test_data import organize_test_files
from .rename_test_files import TestFileOrganizer
import logging

def setup_test_environment():
    """Complete test environment setup"""
    logging.info("Setting up test environment...")
    
    # Setup directories
    dirs = organize_test_files()
    
    # Organize files
    organizer = TestFileOrganizer(
        source_dir=dirs['images'].parent,
        target_dir=dirs['images'].parent
    )
    organizer.run()
    
    logging.info("Test environment setup complete")

if __name__ == "__main__":
    setup_test_environment()