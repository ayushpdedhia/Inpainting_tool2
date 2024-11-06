# src/interface/components/__init__.py
"""
Interface components module for the Image Inpainting Application.
Provides UI components and canvas handling functionality.

Components:
    - UIComponents: Manages UI elements and styling
    - CanvasHandler: Handles canvas operations and mask generation
"""

__version__ = '1.0.0'

from .ui_components import UIComponents
from .canvas_handler import CanvasHandler

__all__ = [
    'UIComponents',
    'CanvasHandler'
]