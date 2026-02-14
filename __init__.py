"""
Mandelbrot Set Visualizer Package

A high-performance, interactive Mandelbrot set explorer using
Pygame for display and Numba for JIT-compiled computation.

Quick Start:
    from mandelbrot import run
    run()

Or from command line:
    python -m mandelbrot

Package Structure:
    - compute.py: JIT-compiled Mandelbrot computation functions
    - colormaps.py: Color scheme definitions (hot, ocean, forest, etc.)
    - renderer.py: Async rendering with caching and prefetching
    - menu.py: Interactive settings UI
    - app.py: Main application and event loop

Controls:
    - Scroll: Zoom in/out at mouse position
    - Drag: Pan around
    - R: Reset to default view
    - ESC: Quit
    - Gear icon: Open settings menu
"""

from .app import run, MandelbrotApp
from .renderer import MandelbrotRenderer
from .colormaps import COLORMAPS, get_colormap, list_colormap_names
from .menu import Menu

__version__ = "1.0.0"
__all__ = [
    "run",
    "MandelbrotApp",
    "MandelbrotRenderer",
    "COLORMAPS",
    "get_colormap",
    "list_colormap_names",
    "Menu",
]
