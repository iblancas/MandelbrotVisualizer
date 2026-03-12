"""
Main application module for the Mandelbrot visualizer.

This module provides the main entry point for running the visualizer.
The actual implementation is split across the app/ subpackage:

- app/core.py: Main MandelbrotApp class
- app/input_handler.py: Mouse and keyboard input processing  
- app/display.py: Display surface management and rendering
- app/image_export.py: High-resolution image export

Usage:
    from MandelbrotVisualizer.app import MandelbrotApp
    app = MandelbrotApp()
    app.run()
    
    # Or use the convenience function:
    from MandelbrotVisualizer.app import run
    run()
"""

# Re-export MandelbrotApp from subpackage for backward compatibility
from .app import MandelbrotApp


def run(width=None, height=None, max_iter=None):
    """
    Run the Mandelbrot visualizer.
    
    Convenience function that creates a MandelbrotApp instance and
    runs it. Handles keyboard interrupts and exceptions gracefully.
    
    Args:
        width: Window width in pixels (default 800)
        height: Window height in pixels (default 800)
        max_iter: Maximum iteration count (default 500)
    """
    app = MandelbrotApp(width, height, max_iter)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        app._cleanup()


# Export for backward compatibility
__all__ = ['MandelbrotApp', 'run']
