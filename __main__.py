"""
Allow running the package directly: python -m MandelbrotVisualizer
Or from inside the folder: python __main__.py
"""
import sys
import os

# When run directly, ensure we can find our modules
if __name__ == "__main__":
    # Add this directory to path for absolute imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from app import run
    run()
