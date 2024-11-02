import pytest
import sys
from pathlib import Path
import logging

# Global variables
TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / 'data' / 'test_samples'
logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_run.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def verify_test_data():
    """Verify test data integrity"""
    required_files = {
        'images': [f'test_image_{i:03d}.JPEG' for i in range(1, 6)],
        'masks': ['mask_large.png', 'mask_edge.png', 'mask_thick.png',
                 'mask_thin.png', 'mask_corner.png', 'mask_small.png']
    }
    
    for category, files in required_files.items():
        dir_path = DATA_DIR / category
        for file in files:
            if not (dir_path / file).exists():
                logger.error(f"Missing required file: {category}/{file}")
                return False
    return True

def run_tests():
    """Run all tests and generate report"""
    setup_logging()
    
    # Verify test data exists
    if not DATA_DIR.exists():
        logger.error(f"Test data directory not found: {DATA_DIR}")
        return 1
    
    if not verify_test_data():
        logger.error("Test data verification failed")
        return 1
    
    logger.info("Starting test run...")
    
    try:
        args = [
            str(TEST_DIR),
            '-v',
            '--cov=src',
            '--cov-report=html',
            '--cov-report=term',
        ]
        
        exit_code = pytest.main(args)
        
        if exit_code == 0:
            logger.info("All tests passed successfully!")
        else:
            logger.error(f"Tests failed with exit code: {exit_code}")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())