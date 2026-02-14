"""
Colormap definitions for Mandelbrot visualization.

Each colormap function returns a numpy array of shape (4096, 3) with
RGB values (uint8). The high resolution (4096 colors) allows for
smooth interpolation in the apply_colormap_smooth function.

To add a new colormap:
1. Define a create_colormap_xxx() function that returns the color array
2. Add it to the COLORMAPS dictionary at the bottom of this file
"""

import numpy as np


NUM_COLORS = 4096  # Resolution of colormap for smooth gradients


def create_colormap_hot():
    """
    Hot colormap: black -> red -> orange -> yellow -> white.
    
    Classic "fire" look with good contrast. Uses a power curve
    to spend more time in the bright colors (glow effect).
    """
    colors = np.zeros((NUM_COLORS, 3), dtype=np.uint8)
    for i in range(NUM_COLORS):
        t_linear = i / (NUM_COLORS - 1)
        t = t_linear ** 0.8  # Power < 1 stretches toward bright colors
        
        colors[i, 0] = int(min(255, 255 * min(1, t * 2.5)))       # Red
        colors[i, 1] = int(min(255, 255 * max(0, (t - 0.4) * 2.5)))  # Green
        colors[i, 2] = int(min(255, 255 * max(0, (t - 0.7) * 3.3)))  # Blue
    return colors


def create_colormap_ocean():
    """
    Ocean colormap: deep blue -> cyan -> white.
    
    Cool, calming color scheme reminiscent of underwater scenes.
    """
    colors = np.zeros((NUM_COLORS, 3), dtype=np.uint8)
    for i in range(NUM_COLORS):
        t = i / (NUM_COLORS - 1)
        colors[i, 0] = int(min(255, 255 * max(0, (t - 0.5) * 2)))  # Red (late)
        colors[i, 1] = int(min(255, 255 * t))                       # Green
        colors[i, 2] = int(min(255, 50 + 205 * t))                  # Blue (starts high)
    return colors


def create_colormap_forest():
    """
    Forest colormap: dark green -> lime -> yellow.
    
    Natural, earthy tones good for organic-looking renders.
    """
    colors = np.zeros((NUM_COLORS, 3), dtype=np.uint8)
    for i in range(NUM_COLORS):
        t = i / (NUM_COLORS - 1)
        colors[i, 0] = int(min(255, 255 * max(0, (t - 0.3) * 1.4)))  # Red
        colors[i, 1] = int(min(255, 80 + 175 * t))                    # Green (high base)
        colors[i, 2] = int(min(255, 255 * max(0, (t - 0.7) * 3.3)))  # Blue (late)
    return colors


def create_colormap_purple():
    """
    Purple colormap: deep purple -> magenta -> pink -> white.
    
    Rich, royal tones with good depth perception.
    """
    colors = np.zeros((NUM_COLORS, 3), dtype=np.uint8)
    for i in range(NUM_COLORS):
        t = i / (NUM_COLORS - 1)
        colors[i, 0] = int(min(255, 100 + 155 * t))                  # Red (high base)
        colors[i, 1] = int(min(255, 255 * max(0, (t - 0.3) * 1.4)))  # Green
        colors[i, 2] = int(min(255, 80 + 175 * t))                    # Blue (high base)
    return colors


def create_colormap_rainbow():
    """
    Rainbow colormap: cycles through hues.
    
    Psychedelic, high-contrast look that shows fine detail.
    Cycles through 5 complete hue rotations.
    """
    colors = np.zeros((NUM_COLORS, 3), dtype=np.uint8)
    for i in range(NUM_COLORS):
        t = i / (NUM_COLORS - 1)
        # HSV to RGB with S=1, V=1
        h = (t * 5) % 1.0  # 5 hue cycles
        
        if h < 1/6:
            colors[i] = [255, int(255 * h * 6), 0]
        elif h < 2/6:
            colors[i] = [int(255 * (2/6 - h) * 6), 255, 0]
        elif h < 3/6:
            colors[i] = [0, 255, int(255 * (h - 2/6) * 6)]
        elif h < 4/6:
            colors[i] = [0, int(255 * (4/6 - h) * 6), 255]
        elif h < 5/6:
            colors[i] = [int(255 * (h - 4/6) * 6), 0, 255]
        else:
            colors[i] = [255, 0, int(255 * (1 - h) * 6)]
    return colors


def create_colormap_grayscale():
    """
    Grayscale colormap: black -> white.
    
    Simple, classic look. Good for seeing raw iteration structure.
    """
    colors = np.zeros((NUM_COLORS, 3), dtype=np.uint8)
    for i in range(NUM_COLORS):
        v = int(255 * i / (NUM_COLORS - 1))
        colors[i] = [v, v, v]
    return colors


# Registry of all available colormaps.
# Keys are display names, values are factory functions.
# Add new colormaps here to make them available in the UI.
COLORMAPS = {
    'Hot': create_colormap_hot,
    'Ocean': create_colormap_ocean,
    'Forest': create_colormap_forest,
    'Purple': create_colormap_purple,
    'Rainbow': create_colormap_rainbow,
    'Grayscale': create_colormap_grayscale,
}


def get_colormap(name):
    """
    Get a colormap by name.
    
    Args:
        name: Key from COLORMAPS dictionary
    
    Returns:
        Colormap array (4096, 3) of uint8 RGB values
    
    Raises:
        KeyError if name not found
    """
    return COLORMAPS[name]()


def get_default_colormap():
    """Get the default colormap (Hot)."""
    return create_colormap_hot()


def list_colormap_names():
    """Get list of available colormap names."""
    return list(COLORMAPS.keys())
