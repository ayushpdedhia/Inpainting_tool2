# tests/__init__.py
import pytest
import logging
import os
import sys
from pathlib import Path
import torch
import warnings

# Configure test environment
TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / 'data'
SAMPLE_DIR = DATA_DIR / 'test_samples'

# Setup logging with more detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TEST_DIR / 'test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create directories if they don't exist
for directory in [DATA_DIR, SAMPLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Make test directories available to all tests
@pytest.fixture(scope="session")
def test_dir():
    return TEST_DIR

@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR

@pytest.fixture(scope="session")
def sample_dir():
    return SAMPLE_DIR

# Device configuration fixture
@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Skip GPU tests if no GPU available
def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

# Filter specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="streamlit")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated")

# Environment validation
def validate_environment():
    """Validate test environment setup"""
    try:
        assert torch.__version__ >= "1.7.0", "Torch version should be >= 1.7.0"
        assert sys.version_info >= (3, 7), "Python version should be >= 3.7"
        logger.info("Environment validation passed")
        return True
    except AssertionError as e:
        logger.error(f"Environment validation failed: {str(e)}")
        return False

# Run environment validation
validate_environment()