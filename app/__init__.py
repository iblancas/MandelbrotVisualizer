"""
Application package for the Mandelbrot visualizer.

This package contains the main application class and supporting modules
for input handling, display management, and image export.

Modules:
- core: Main MandelbrotApp class
- input_handler: Mouse and keyboard input processing
- display: Display surface management and rendering
- image_export: High-resolution image export functionality

Usage:
    from MandelbrotVisualizer.app import MandelbrotApp
    
    app = MandelbrotApp(width=800, height=800, max_iter=500)
    app.run()
"""

from app.core import MandelbrotApp


def run():
    """Convenience function to create and run the application."""
    app = MandelbrotApp()
    app.run()


__all__ = ['MandelbrotApp', 'run']
