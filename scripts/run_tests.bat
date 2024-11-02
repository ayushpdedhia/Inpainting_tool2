@echo off
echo Setting up test environment...
python -m src.utils.manage_test_data

echo Running tests...
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

echo Test run complete!
pause