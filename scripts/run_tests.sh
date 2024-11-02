#!/bin/bash
# scripts/run_tests.sh

echo "Setting up test environment..."
python -m src.utils.manage_test_data  # This will handle both organize and rename

echo "Running tests..."
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

echo "Test run complete!"