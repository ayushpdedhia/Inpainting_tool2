# src/interface/__init__.py
"""
Interface module for the Image Inpainting Application.
Provides the main application class and entry point.
"""

__version__ = '1.0.0'

from .app import InpaintingApp, main

__all__ = [
    'InpaintingApp',
    'main'
]