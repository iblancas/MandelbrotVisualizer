"""
Input handling module for the Mandelbrot visualizer.

This module contains the InputHandler class which manages all user input
including mouse interactions (zoom, pan, drag) and keyboard shortcuts.

The input handler abstracts away the complexity of input processing,
allowing the main application to focus on rendering and display logic.

Key Features:
- Mouse wheel zoom with focus point preservation
- Click-and-drag panning in the complex plane
- Keyboard shortcuts (R to reset, Escape to quit, Cmd+S to save)
- Coordinate transformation between screen and complex plane
"""

import pygame


class InputHandler:
    """
    Handles all user input for the Mandelbrot visualizer.
    
    This class processes mouse and keyboard events, translating them
    into operations on the complex plane viewing window. It maintains
    state for drag operations and provides clean interfaces for the
    main application to respond to user actions.
    
    Attributes:
        width: Window width in pixels
        height: Window height in pixels
        dragging: Whether a drag operation is in progress
        drag_start: Screen coordinates where drag began
        drag_start_bounds: Complex plane bounds when drag began
        
    Coordinate Systems:
        Screen coordinates: (0,0) at top-left, y increases downward
        Complex plane: Standard mathematical coordinates, y increases upward
    """
    
    # Zoom factors - determines how fast we zoom in/out per scroll step
    # Values < 1 zoom in (shrink view), values > 1 zoom out (expand view)
    ZOOM_IN_FACTOR = 0.85   # 15% zoom in per scroll up
    ZOOM_OUT_FACTOR = 1.18  # 18% zoom out per scroll down (slightly asymmetric for feel)
    
    def __init__(self, width, height):
        """
        Initialize the input handler.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
        """
        self.width = width
        self.height = height
        
        # Drag state tracking
        self.dragging = False
        self.drag_start = None
        self.drag_start_bounds = None
    
    def screen_to_complex(self, screen_x, screen_y, bounds):
        """
        Convert screen coordinates to complex plane coordinates.
        
        The complex plane has y increasing upward, while screen coordinates
        have y increasing downward, so we need to flip the y axis.
        
        Args:
            screen_x: X position in screen coordinates (pixels from left)
            screen_y: Y position in screen coordinates (pixels from top)
            bounds: Tuple of (x_min, x_max, y_min, y_max) in complex plane
            
        Returns:
            Tuple of (real, imag) coordinates in the complex plane
        """
        x_min, x_max, y_min, y_max = bounds
        
        # Linear interpolation from screen to complex coordinates
        real = x_min + (x_max - x_min) * screen_x / self.width
        # Note: y is flipped because screen y=0 is top, complex y_max is top
        imag = y_min + (y_max - y_min) * (self.height - screen_y) / self.height
        
        return real, imag
    
    def handle_zoom(self, event, bounds, menu_check_func):
        """
        Handle mouse wheel zoom events.
        
        Zooms centered on the mouse position, preserving the point under
        the cursor. This creates intuitive "zoom to cursor" behavior.
        
        The zoom algorithm:
        1. Convert mouse position to complex coordinates (this is the focus point)
        2. Calculate new view dimensions based on zoom factor
        3. Position the new view so the focus point stays under the cursor
        
        Args:
            event: Pygame MOUSEWHEEL event
            bounds: Current (x_min, x_max, y_min, y_max) bounds
            menu_check_func: Function to check if mouse is over menu
            
        Returns:
            New bounds tuple if zoom occurred, None if zoom was blocked (e.g., mouse over menu)
        """
        # Don't zoom when mouse is over the menu
        mouse_pos = pygame.mouse.get_pos()
        if menu_check_func(mouse_pos):
            return None
        
        x_min, x_max, y_min, y_max = bounds
        mx, my = mouse_pos
        
        # Get the complex coordinate under the mouse - this will be our zoom focus
        cx, cy = self.screen_to_complex(mx, my, bounds)
        
        # Determine zoom direction: scroll up = zoom in, scroll down = zoom out
        zoom_factor = self.ZOOM_IN_FACTOR if event.y > 0 else self.ZOOM_OUT_FACTOR
        
        # Calculate new view dimensions
        new_width = (x_max - x_min) * zoom_factor
        new_height = (y_max - y_min) * zoom_factor
        
        # Calculate where the focus point is relative to the current view (0 to 1)
        x_ratio = (cx - x_min) / (x_max - x_min)
        y_ratio = (cy - y_min) / (y_max - y_min)
        
        # Position new view so focus point maintains same relative position
        new_x_min = cx - x_ratio * new_width
        new_x_max = cx + (1 - x_ratio) * new_width
        new_y_min = cy - y_ratio * new_height
        new_y_max = cy + (1 - y_ratio) * new_height
        
        return (new_x_min, new_x_max, new_y_min, new_y_max)
    
    def handle_mouse_down(self, event, bounds, menu_check_func):
        """
        Handle mouse button press events.
        
        Left click outside the menu starts a drag operation for panning.
        We store the initial mouse position and view bounds to calculate
        the pan offset during drag.
        
        Args:
            event: Pygame MOUSEBUTTONDOWN event
            bounds: Current viewing bounds in complex plane
            menu_check_func: Function to check if position is over menu
            
        Returns:
            True if drag was started, False otherwise
        """
        if event.button == 1:  # Left mouse button
            if not menu_check_func(event.pos):
                self.dragging = True
                self.drag_start = pygame.mouse.get_pos()
                self.drag_start_bounds = bounds
                return True
        return False
    
    def handle_mouse_up(self, event):
        """
        Handle mouse button release events.
        
        Ends any ongoing drag operation.
        
        Args:
            event: Pygame MOUSEBUTTONUP event
            
        Returns:
            True if a drag was ended, False otherwise
        """
        if event.button == 1:  # Left mouse button
            was_dragging = self.dragging
            self.dragging = False
            return was_dragging
        return False
    
    def handle_mouse_motion(self, bounds):
        """
        Handle mouse motion during drag operations.
        
        Calculates the pan offset in complex plane units based on how far
        the mouse has moved from the drag start position.
        
        The pan is calculated relative to the bounds at drag start,
        not the current bounds, which prevents accumulated drift.
        
        Args:
            bounds: Current viewing bounds (used only if not dragging)
            
        Returns:
            New bounds tuple if dragging, None otherwise
        """
        if not self.dragging or not self.drag_start:
            return None
        
        mx, my = pygame.mouse.get_pos()
        start_mx, start_my = self.drag_start
        x_min, x_max, y_min, y_max = self.drag_start_bounds
        
        # Calculate drag distance in complex plane units
        # Note: we subtract current from start (inverse direction for natural feel)
        dx = (start_mx - mx) / self.width * (x_max - x_min)
        # Y is not inverted because screen y-down cancels with complex y-up
        dy = (my - start_my) / self.height * (y_max - y_min)
        
        # Apply offset to original drag-start bounds
        new_x_min = x_min + dx
        new_x_max = x_max + dx
        new_y_min = y_min + dy
        new_y_max = y_max + dy
        
        return (new_x_min, new_x_max, new_y_min, new_y_max)
    
    def handle_key(self, event, default_bounds):
        """
        Handle keyboard events.
        
        Supported keys:
        - R: Reset view to default bounds
        - Escape: Request application quit
        - Cmd+S (Mac) / Ctrl+S (Windows/Linux): Save high-res image
        
        Args:
            event: Pygame KEYDOWN event
            default_bounds: Default viewing bounds for reset
            
        Returns:
            Dict with action results:
            - 'reset': New bounds if reset was requested
            - 'quit': True if quit was requested
            - 'save': True if save was requested
        """
        result = {'reset': None, 'quit': False, 'save': False}
        
        if event.key == pygame.K_r:
            # Reset to default view (full Mandelbrot set overview)
            result['reset'] = default_bounds
            
        elif event.key == pygame.K_ESCAPE:
            # Request application quit
            result['quit'] = True
            
        elif event.key == pygame.K_s:
            # Check for Cmd (Mac) or Ctrl (Windows/Linux) modifier
            mods = pygame.key.get_mods()
            if mods & pygame.KMOD_META or mods & pygame.KMOD_CTRL:
                result['save'] = True
        
        return result
