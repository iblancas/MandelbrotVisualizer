"""
Image export module for the Mandelbrot visualizer.

This module handles exporting high-resolution images of the current view.
It supports both CPU and GPU rendering paths and applies supersampling
for high-quality antialiased output.

Export Features:
- 4x resolution multiplier for detailed output
- 2x supersampling for smooth edges
- Automatic GPU/CPU selection based on availability
- Custom formula support for user-defined functions
- Timestamped filenames saved to desktop
"""

import os
from datetime import datetime
import numpy as np
import pygame


class ImageExporter:
    """
    Handles exporting high-resolution fractal images.
    
    This class encapsulates the logic for rendering and saving high-resolution
    images of the current view. It uses the same rendering pipeline as the
    main display but at higher resolution with supersampling.
    
    The export process:
    1. Calculate high-resolution dimensions (4x base resolution)
    2. Add 2x supersampling for antialiasing
    3. Compute fractal data at full resolution
    4. Apply colormap
    5. Downscale with averaging for smooth result
    6. Save as PNG to desktop
    
    Attributes:
        base_width: Base window width for calculating export size
        base_height: Base window height for calculating export size
    """
    
    # Resolution multiplier for high-res export (4x = 3200x3200 from 800x800)
    RESOLUTION_SCALE = 4
    
    # Supersampling factor for antialiasing
    SUPERSAMPLING = 2
    
    def __init__(self, base_width, base_height):
        """
        Initialize the image exporter.
        
        Args:
            base_width: Base window width in pixels
            base_height: Base window height in pixels
        """
        self.base_width = base_width
        self.base_height = base_height
    
    def export(self, bounds, renderer, compute_funcs, display_manager=None):
        """
        Export a high-resolution image of the current view.
        
        This method renders the fractal at high resolution and saves it
        as a PNG file to the user's desktop.
        
        Args:
            bounds: Tuple of (x_min, x_max, y_min, y_max) - current view bounds
            renderer: MandelbrotRenderer instance for settings and GPU access
            compute_funcs: Dict of compute functions:
                - 'compute': Standard compute_mandelbrot function
                - 'compute_custom': Custom formula compute function
                - 'colormap': apply_colormap_smooth function
                - 'downscale': downscale_2x function
            display_manager: Optional DisplayManager to show progress
            
        Returns:
            Path to saved file, or None if export failed
        """
        x_min, x_max, y_min, y_max = bounds
        
        # Calculate export dimensions
        hi_width = self.base_width * self.RESOLUTION_SCALE
        hi_height = self.base_height * self.RESOLUTION_SCALE
        
        # Apply supersampling (render at 2x then downscale)
        render_width = hi_width * self.SUPERSAMPLING
        render_height = hi_height * self.SUPERSAMPLING
        
        # Show progress if display manager provided
        if display_manager:
            display_manager.set_title("Saving high-res image... (this may take a moment)")
            pygame.display.flip()
        
        # Extract compute functions
        compute_mandelbrot = compute_funcs['compute']
        compute_mandelbrot_custom = compute_funcs['compute_custom']
        apply_colormap_smooth = compute_funcs['colormap']
        downscale_2x = compute_funcs['downscale']
        
        # Compute fractal data at high resolution
        data = self._compute_high_res(
            bounds, render_width, render_height, renderer,
            compute_mandelbrot, compute_mandelbrot_custom
        )
        
        # Apply colormap to get RGB image
        hi_rgb = np.empty((render_height, render_width, 3), dtype=np.uint8)
        self._apply_colormap(data, renderer, apply_colormap_smooth, hi_rgb)
        
        # Downscale with 2x supersampling for smooth result
        final_rgb = np.empty((hi_height, hi_width, 3), dtype=np.uint8)
        self._downscale(hi_rgb, final_rgb, renderer, downscale_2x)
        
        # Flip for correct orientation (numpy y=0 at top, we want bottom)
        final_rgb = np.flipud(final_rgb)
        
        # Save to file
        filename = self._save_image(final_rgb)
        
        # Update title with result
        if display_manager:
            display_manager.set_title(f"Saved: {os.path.basename(filename)} - Mandelbrot Set")
        
        print(f"High-resolution image saved to: {filename}")
        return filename
    
    def _compute_high_res(self, bounds, width, height, renderer,
                          compute_mandelbrot, compute_mandelbrot_custom):
        """
        Compute fractal data at high resolution.
        
        Selects the appropriate compute method based on:
        1. Custom formula (CPU only, user-defined function)
        2. GPU if available and enabled
        3. CPU with Numba JIT as fallback
        
        Args:
            bounds: Viewing bounds in complex plane
            width: Render width in pixels
            height: Render height in pixels
            renderer: Renderer for settings access
            compute_mandelbrot: Standard compute function
            compute_mandelbrot_custom: Custom formula compute function
            
        Returns:
            NumPy array of iteration counts / escape values
        """
        x_min, x_max, y_min, y_max = bounds
        
        if renderer._prepared_formula is not None:
            # Custom formula - must use CPU (Python-based iteration)
            return compute_mandelbrot_custom(
                x_min, x_max, y_min, y_max,
                width, height,
                renderer.max_iter,
                renderer._prepared_formula,
                renderer.escape_radius,
                renderer.julia_mode,
                renderer.julia_c_real,
                renderer.julia_c_imag
            )
        elif renderer.use_gpu and renderer._gpu_compute is not None:
            # GPU acceleration available - use PyTorch
            return renderer._gpu_compute.compute_mandelbrot(
                x_min, x_max, y_min, y_max,
                width, height,
                renderer.max_iter,
                renderer.func_id,
                renderer.escape_radius,
                renderer.julia_mode,
                renderer.julia_c_real,
                renderer.julia_c_imag
            )
        else:
            # CPU with Numba JIT compilation
            return compute_mandelbrot(
                x_min, x_max, y_min, y_max,
                width, height,
                renderer.max_iter,
                func_id=renderer.func_id,
                escape_radius=renderer.escape_radius,
                julia_mode=renderer.julia_mode,
                julia_c_real=renderer.julia_c_real,
                julia_c_imag=renderer.julia_c_imag
            )
    
    def _apply_colormap(self, data, renderer, apply_colormap_smooth, output):
        """
        Apply colormap to iteration data.
        
        Uses GPU if available for faster processing of large images.
        
        Args:
            data: Iteration count data array
            renderer: Renderer for GPU access
            apply_colormap_smooth: CPU colormap function
            output: Pre-allocated output array for RGB data
        """
        if renderer.use_gpu and renderer._gpu_compute is not None:
            renderer._gpu_compute.apply_colormap_smooth(
                data, renderer.max_iter, renderer.colormap, output
            )
        else:
            apply_colormap_smooth(
                data, renderer.max_iter, renderer.colormap, output
            )
    
    def _downscale(self, hi_rgb, output, renderer, downscale_2x):
        """
        Downscale high-resolution image with antialiasing.
        
        Uses 2x2 box filter averaging for smooth results.
        
        Args:
            hi_rgb: High-resolution RGB array
            output: Pre-allocated output array
            renderer: Renderer for GPU access
            downscale_2x: CPU downscale function
        """
        if renderer.use_gpu and renderer._gpu_compute is not None:
            renderer._gpu_compute.downscale_2x(hi_rgb, output)
        else:
            downscale_2x(hi_rgb, output)
    
    def _save_image(self, rgb_data):
        """
        Save RGB data as PNG file to desktop.
        
        Generates a timestamped filename to avoid overwrites.
        
        Args:
            rgb_data: NumPy array of shape (height, width, 3)
            
        Returns:
            Full path to saved file
        """
        # Create pygame surface from RGB data
        surface = pygame.surfarray.make_surface(rgb_data.swapaxes(0, 1))
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        desktop_path = os.path.expanduser("~/Desktop")
        filename = os.path.join(desktop_path, f"mandelbrot_{timestamp}.png")
        
        # Save as PNG
        pygame.image.save(surface, filename)
        
        return filename
