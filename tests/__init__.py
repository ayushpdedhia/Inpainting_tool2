# tests/__init__.py
import pytest
import logging
import os
from pathlib import Path
import torch

# Configure test environment
TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / 'data'
SAMPLE_DIR = DATA_DIR / 'test_samples'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

# Skip GPU tests if no GPU available
def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)