# tests/__init__.py
import pytest
import logging
import os
from pathlib import Path

# Configure test environment
TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / 'data'
SAMPLE_DIR = DATA_DIR / 'test_samples'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Pytest configuration
def pytest_configure(config):
    """Setup pytest configuration"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )