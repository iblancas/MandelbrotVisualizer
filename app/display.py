"""
Display management module for the Mandelbrot visualizer.

This module contains the DisplayManager class which handles all rendering
and display operations, including surface creation, coordinate transformation,
and blitting rendered images to the screen.

The display manager handles the complexity of mapping rendered fractal data
(which may have different bounds than the current view) to screen coordinates,
enabling smooth panning and zooming with cached render data.

Key Responsibilities:
- Managing the pygame display surface
- Transforming and blitting rendered surfaces to match current view
- Handling render history for smooth zoom-out transitions
- Surface lifecycle management
"""

import pygame
import numpy as np


class DisplayManager:
    """
    Manages display rendering for the Mandelbrot visualizer.
    
    This class handles the transformation of rendered fractal images
    to screen coordinates, supporting smooth panning and zooming by
    reusing cached renders that may not exactly match the current view.
    
    The key challenge this class solves is coordinate mapping:
    - Rendered images have specific bounds in the complex plane
    - The current view may have different bounds (during pan/zoom)
    - We need to extract and scale the overlapping region correctly
    
    Attributes:
        width: Display width in pixels
        height: Display height in pixels
        screen: Pygame display surface
        clock: Pygame clock for framerate control
        render_history: List of (surface, bounds) for smooth transitions
        max_history: Maximum number of historical renders to keep
    """
    
    # Number of previous renders to keep for zoom-out transitions
    # More history = smoother zoom-out but more memory usage
    MAX_HISTORY = 5
    
    def __init__(self, width, height):
        """
        Initialize the display manager.
        
        Note: This only stores dimensions. Call init_pygame() separately
        to actually create the display, as pygame initialization may need
        to happen at a specific time in the application lifecycle.
        
        Args:
            width: Display width in pixels
            height: Display height in pixels
        """
        self.width = width
        self.height = height
        
        # Pygame objects (initialized in init_pygame)
        self.screen = None
        self.clock = None
        
        # Current display state
        self.current_surface = None
        self.current_rgb = None
        self.render_bounds = None
        
        # Render history for smooth zoom-out
        self.render_history = []
    
    def init_pygame(self, title="Mandelbrot Set"):
        """
        Initialize pygame and create the display window.
        
        Args:
            title: Initial window title
            
        Returns:
            The pygame screen surface
        """
        pygame.init()
        
        # Create double-buffered display for smooth rendering
        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.DOUBLEBUF
        )
        pygame.display.set_caption(title)
        
        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
        
        return self.screen
    
    def set_title(self, title):
        """
        Set the window title.
        
        Args:
            title: New window title string
        """
        pygame.display.set_caption(title)
    
    def update_surface(self, rgb_data, bounds):
        """
        Update the current display surface with new render data.
        
        Creates a new pygame surface from the RGB data and adds the
        previous surface to the history for smooth transitions.
        
        Args:
            rgb_data: NumPy array of shape (height, width, 3) with RGB values
            bounds: Tuple of (x_min, x_max, y_min, y_max) for this render
        """
        self.current_rgb = rgb_data
        self.render_bounds = bounds
        
        # Create pygame surface from RGB data
        # Note: swapaxes because pygame expects (width, height, 3) not (height, width, 3)
        self.current_surface = pygame.surfarray.make_surface(
            rgb_data.swapaxes(0, 1)
        )
        
        # Add to history for smooth zoom-out
        self.render_history.append((self.current_surface.copy(), bounds))
        if len(self.render_history) > self.MAX_HISTORY:
            self.render_history.pop(0)
    
    def clear_history(self):
        """
        Clear the render history.
        
        Should be called when settings change (colormap, iterations, etc.)
        that invalidate previous renders.
        """
        self.render_history.clear()
    
    def draw_frame(self, current_bounds, menu):
        """
        Draw a complete frame to the screen.
        
        This draws the render history (oldest first) then the current
        surface, allowing smooth transitions during zoom-out.
        
        Args:
            current_bounds: Current viewing bounds in complex plane
            menu: Menu object to draw on top
        """
        # Clear screen to black
        self.screen.fill((0, 0, 0))
        
        # Draw historical renders first (oldest to newest)
        # This provides smooth transitions during zoom-out
        for hist_surface, hist_bounds in self.render_history:
            self._blit_surface_to_view(hist_surface, hist_bounds, current_bounds)
        
        # Draw current surface on top
        if self.current_surface is not None:
            self._blit_surface_to_view(
                self.current_surface, 
                self.render_bounds, 
                current_bounds
            )
        
        # Draw menu overlay
        menu.draw(self.screen)
        
        # Flip double buffer to display
        pygame.display.flip()
    
    def _blit_surface_to_view(self, surface, render_bounds, view_bounds):
        """
        Blit a rendered surface to the screen, transforming for current view.
        
        This is the core coordinate transformation method. It handles the case
        where the rendered surface has different bounds than the current view,
        extracting and scaling the overlapping region appropriately.
        
        Algorithm:
        1. Calculate where the current view maps to in the rendered surface
        2. Clamp to valid surface bounds (in case view extends beyond render)
        3. Scale the extracted region to fill the appropriate screen area
        
        Args:
            surface: Pygame surface containing rendered fractal
            render_bounds: (x_min, x_max, y_min, y_max) bounds of the render
            view_bounds: (x_min, x_max, y_min, y_max) bounds of current view
        """
        if surface is None:
            return
            
        surf_w = surface.get_width()
        surf_h = surface.get_height()
        
        # Unpack bounds
        rx_min, rx_max, ry_min, ry_max = render_bounds
        vx_min, vx_max, vy_min, vy_max = view_bounds
        
        render_w = rx_max - rx_min
        render_h = ry_max - ry_min
        
        # Avoid division by zero
        if render_w <= 0 or render_h <= 0:
            return
        
        # Map current view coordinates to pixel positions in the rendered surface
        # These may be outside [0, surf_w/surf_h] if view extends beyond render
        src_left = (vx_min - rx_min) / render_w * surf_w
        src_right = (vx_max - rx_min) / render_w * surf_w
        src_bottom = (vy_min - ry_min) / render_h * surf_h
        src_top = (vy_max - ry_min) / render_h * surf_h
        
        # Clamp source rectangle to valid surface bounds
        src_left_c = max(0, min(surf_w, src_left))
        src_right_c = max(0, min(surf_w, src_right))
        src_bottom_c = max(0, min(surf_h, src_bottom))
        src_top_c = max(0, min(surf_h, src_top))
        
        # Calculate source region dimensions
        src_w = src_right_c - src_left_c
        src_h = src_top_c - src_bottom_c
        
        if src_w <= 0 or src_h <= 0:
            return  # No overlap between render and view
        
        # View dimensions (may be different from source due to clamping)
        view_w = src_right - src_left
        view_h = src_top - src_bottom
        
        if view_w <= 0 or view_h <= 0:
            return
        
        # Calculate destination rectangle on screen
        # Account for any clamping that occurred
        dst_left = (src_left_c - src_left) / view_w * self.width
        dst_right = self.width - (src_right - src_right_c) / view_w * self.width
        dst_bottom = (src_bottom_c - src_bottom) / view_h * self.height
        dst_top = self.height - (src_top - src_top_c) / view_h * self.height
        
        dst_w = dst_right - dst_left
        dst_h = dst_top - dst_bottom
        
        if dst_w <= 0 or dst_h <= 0:
            return
        
        try:
            # Extract the relevant portion of the surface
            # Note: pygame surface has y=0 at top, so we transform coordinates
            src_rect = pygame.Rect(
                int(src_left_c),
                int(surf_h - src_top_c),  # Flip y for pygame coordinates
                int(src_w),
                int(src_h)
            )
            subsurface = surface.subsurface(src_rect)
            
            # Scale to destination size using smooth scaling for quality
            scaled = pygame.transform.smoothscale(
                subsurface, 
                (int(dst_w), int(dst_h))
            )
            
            # Blit to screen at destination position
            self.screen.blit(scaled, (int(dst_left), int(self.height - dst_top)))
            
        except ValueError:
            # Subsurface rectangle out of bounds - skip this frame
            # This can happen during rapid pan/zoom transitions
            pass
    
    def tick(self, fps=60):
        """
        Maintain frame rate and return time since last tick.
        
        Args:
            fps: Target frames per second
            
        Returns:
            Time in milliseconds since last tick
        """
        return self.clock.tick(fps)
    
    def cleanup(self):
        """
        Clean up display resources.
        
        Clears surfaces and attempts to quit pygame cleanly.
        """
        self.current_surface = None
        self.current_rgb = None
        self.render_history.clear()
        
        try:
            pygame.quit()
        except Exception:
            pass  # pygame may already be quit or in bad state
