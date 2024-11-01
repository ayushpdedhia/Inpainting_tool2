import pytest
import sys
from pathlib import Path
import logging

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

def run_tests():
    """Run all tests and generate report"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Get test directory
    test_dir = Path(__file__).parent / 'tests'
    
    # Verify test data exists
    data_dir = Path(__file__).parent / 'data' / 'test_samples'
    if not data_dir.exists():
        logger.error(f"Test data directory not found: {data_dir}")
        return 1
        
    required_dirs = ['images', 'masks']
    for dir_name in required_dirs:
        if not (data_dir / dir_name).exists():
            logger.error(f"Required directory not found: {data_dir / dir_name}")
            return 1
    
    logger.info("Starting test run...")
    
    try:
        # Run pytest with coverage
        args = [
            str(test_dir),
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