"""
Allow running the package directly: python -m MandelbrotVisualizer
Or from inside the folder: python __main__.py
"""
import sys
import os

# Handle both direct execution and module execution
if __name__ == "__main__":
    # When run directly, add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from MandelbrotVisualizer.app import run
    run()
else:
    # When run as module (python -m MandelbrotVisualizer)
    from .app import run
    run()
