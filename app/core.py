"""
Core application module for the Mandelbrot visualizer.

This module contains the main MandelbrotApp class which orchestrates
all components of the visualizer including rendering, display, input
handling, and the settings menu.

Architecture:
- InputHandler: Processes mouse and keyboard events
- DisplayManager: Handles pygame display and surface blitting
- ImageExporter: Exports high-resolution images
- MandelbrotRenderer: Async fractal computation with caching
- Menu: Settings UI overlay

The application uses an async rendering model where fractal computation
happens in background threads while the UI remains responsive.
"""

import atexit
import pygame

from app.input_handler import InputHandler
from app.display import DisplayManager
from app.image_export import ImageExporter


class MandelbrotApp:
    """
    Main application class for the Mandelbrot visualizer.
    
    Coordinates between rendering, display, input handling, and settings.
    Uses component-based architecture for maintainability.
    
    The main loop:
    1. Handle input events (zoom, pan, keyboard)
    2. Check for menu requests (save, GPU toggle)
    3. Check for completed async renders
    4. Start new renders if needed (with debouncing)
    5. Draw current frame
    
    Attributes:
        width: Window width in pixels
        height: Window height in pixels
        max_iter: Maximum iteration count for fractal computation
        bounds: Current viewing bounds as (x_min, x_max, y_min, y_max)
        running: Whether the main loop should continue
    """
    
    # Default configuration
    DEFAULT_VIZ_SIZE = 800
    DEFAULT_MAX_ITER = 500
    SIDEBAR_WIDTH = 280
    
    # Default view bounds - shows classic Mandelbrot overview
    # Positioned to show the full set with some margin
    DEFAULT_BOUNDS = (-2.5, 1.0, -1.75, 1.75)
    
    # Render debounce delay in milliseconds
    # Prevents starting new renders while user is actively panning/zooming
    RENDER_DELAY_MS = 25
    
    def __init__(self, viz_size=None, max_iter=None):
        """
        Initialize the application.
        
        Does not create pygame window - call run() to start the application.
        
        Args:
            viz_size: Visualization area size in pixels (square, default 800)
            max_iter: Maximum iteration count (default 500)
        """
        # Apply defaults
        self.viz_size = viz_size or self.DEFAULT_VIZ_SIZE
        self.max_iter = max_iter or self.DEFAULT_MAX_ITER
        
        # Window dimensions (viz area + sidebar)
        self.width = self.viz_size + self.SIDEBAR_WIDTH
        self.height = self.viz_size
        
        # Current viewing bounds in complex plane
        self.x_min, self.x_max, self.y_min, self.y_max = self.DEFAULT_BOUNDS
        
        # Components (initialized in run)
        self.input_handler = None
        self.display = None
        self.exporter = None
        self.renderer = None
        self.menu = None
        
        # Render timing for debouncing
        self.last_action_time = 0
        self.pending_render = False
        
        # Application state
        self.running = False
        self._cleanup_registered = False
        self._cleaned_up = False
    
    @property
    def bounds(self):
        """Get current viewing bounds as tuple."""
        return (self.x_min, self.x_max, self.y_min, self.y_max)
    
    @bounds.setter
    def bounds(self, value):
        """Set viewing bounds from tuple."""
        self.x_min, self.x_max, self.y_min, self.y_max = value
    
    def run(self):
        """
        Run the application main loop.
        
        Initializes all components, performs warmup, then enters the
        main event loop until quit is requested.
        """
        # Initialize components
        self._init_components()
        self._warmup_and_initial_render()
        
        # Register cleanup handler for graceful shutdown
        if not self._cleanup_registered:
            atexit.register(self._cleanup)
            self._cleanup_registered = True
        
        # Main event loop
        self.running = True
        while self.running:
            current_time = pygame.time.get_ticks()
            
            # Process events
            self._handle_events(current_time)
            
            # Handle menu requests
            self._handle_menu_requests(current_time)
            
            # Check for completed async renders
            self._check_render_result()
            
            # Start new render if needed (with debouncing)
            self._maybe_start_render(current_time)
            
            # Draw current frame
            self.display.draw_frame(self.bounds, self.menu)
            
            # Maintain 60 FPS
            self.display.tick(60)
        
        # Clean up on exit
        self._cleanup()
    
    def _init_components(self):
        """
        Initialize all application components.
        
        Creates input handler, display manager, renderer, menu,
        and image exporter.
        """
        # Import here to avoid circular imports
        from renderer import MandelbrotRenderer
        from menu import Menu
        
        # Create components using visualization size
        self.input_handler = InputHandler(self.viz_size, self.viz_size)
        self.display = DisplayManager(self.viz_size, self.SIDEBAR_WIDTH)
        self.exporter = ImageExporter(self.viz_size, self.viz_size)
        
        # Initialize pygame via display manager
        self.display.init_pygame("Mandelbrot Set - Scroll to zoom, drag to pan")
        
        # Create renderer with visualization dimensions
        self.renderer = MandelbrotRenderer(
            self.viz_size, self.viz_size, self.max_iter, 
            use_gpu=None
        )
        
        # Create menu in sidebar (positioned at start of sidebar)
        self.menu = Menu(
            x=self.viz_size + 10, 
            y=10, 
            width=self.SIDEBAR_WIDTH - 20,
            screen_width=self.width,
            screen_height=self.height
        )
        self.menu.max_iter = self.max_iter
        
        # Update menu with GPU status
        self._update_menu_gpu_status()
    
    def _warmup_and_initial_render(self):
        """
        Perform JIT/GPU warmup and render initial view.
        
        The first render triggers Numba JIT compilation which can take
        several seconds. We show a message during this time.
        """
        from compute import (
            compute_mandelbrot, apply_colormap_smooth, 
            downscale_2x, warmup_jit
        )
        import numpy as np
        
        # Warm up Numba JIT compilation
        self.display.set_title("Compiling (first run only)...")
        warmup_jit(self.renderer.colormap)
        
        # Warm up GPU if being used
        if self.renderer.use_gpu and self.renderer._gpu_compute:
            self.display.set_title("Warming up GPU...")
            self.renderer._gpu_compute.warmup(self.renderer.colormap)
        
        # Calculate initial bounds with margin for smoother panning
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min
        margin_x = w * self.renderer.margin
        margin_y = h * self.renderer.margin
        
        init_bounds = (
            self.x_min - margin_x, self.x_max + margin_x,
            self.y_min - margin_y, self.y_max + margin_y
        )
        
        # Compute initial fractal data
        if self.renderer.use_gpu and self.renderer._gpu_compute:
            data = self.renderer._gpu_compute.compute_mandelbrot(
                init_bounds[0], init_bounds[1], 
                init_bounds[2], init_bounds[3],
                self.renderer.render_width, 
                self.renderer.render_height,
                self.max_iter
            )
        else:
            data = compute_mandelbrot(
                init_bounds[0], init_bounds[1], 
                init_bounds[2], init_bounds[3],
                self.renderer.render_width, 
                self.renderer.render_height,
                self.max_iter
            )
        
        # Apply colormap and downscale
        apply_colormap_smooth(
            data, self.max_iter, 
            self.renderer.colormap, 
            self.renderer.rgb_hi
        )
        downscale_2x(self.renderer.rgb_hi, self.renderer.rgb)
        
        # Flip for pygame coordinate system
        rgb_flipped = np.flipud(self.renderer.rgb).copy()
        
        # Initialize renderer cache
        self.renderer.data_cache[:] = data
        self.renderer.cache_bounds = init_bounds
        self.renderer.actual_bounds = init_bounds
        
        # Start background prefetching
        self.renderer._start_prefetch(init_bounds)
        
        # Update display
        self.display.update_surface(rgb_flipped, init_bounds)
        self.display.draw_frame(self.bounds, self.menu)
        
        # Update title with GPU status
        gpu_status = "GPU" if self.renderer.get_gpu_info()['enabled'] else "CPU"
        self.display.set_title(
            f"Mandelbrot Set [{gpu_status}] - Scroll to zoom, drag to pan, R to reset"
        )
    
    def _handle_events(self, current_time):
        """
        Process all pending pygame events.
        
        Delegates to input handler and menu, tracking which events
        require a re-render.
        
        Args:
            current_time: Current pygame tick time in milliseconds
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            
            # Let menu handle event first
            menu_handled, need_recompute = self.menu.handle_event(event)
            if need_recompute:
                self._apply_menu_settings(current_time)
            if menu_handled:
                continue
            
            # Handle input events
            if event.type == pygame.MOUSEWHEEL:
                new_bounds = self.input_handler.handle_zoom(
                    event, self.bounds, self.menu.point_in_menu
                )
                if new_bounds:
                    self.bounds = new_bounds
                    self._trigger_render(current_time)
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.input_handler.handle_mouse_down(
                    event, self.bounds, self.menu.point_in_menu
                )
                
            elif event.type == pygame.MOUSEBUTTONUP:
                if self.input_handler.handle_mouse_up(event):
                    self._trigger_render(current_time)
                    
            elif event.type == pygame.MOUSEMOTION:
                new_bounds = self.input_handler.handle_mouse_motion(self.bounds)
                if new_bounds:
                    self.bounds = new_bounds
                    
            elif event.type == pygame.KEYDOWN:
                result = self.input_handler.handle_key(event, self.DEFAULT_BOUNDS)
                if result['reset']:
                    self.bounds = result['reset']
                    self.display.clear_history()
                    self._trigger_render(current_time)
                if result['quit']:
                    self.running = False
                if result['save']:
                    self._save_high_res_image()
    
    def _handle_menu_requests(self, current_time):
        """
        Handle special requests from the menu.
        
        Args:
            current_time: Current pygame tick time
        """
        # Handle save request
        if self.menu.save_requested:
            self.menu.save_requested = False
            self._save_high_res_image()
        
        # Handle GPU toggle request
        if self.menu.gpu_toggle_requested:
            self.menu.gpu_toggle_requested = False
            self._handle_gpu_toggle(current_time)
    
    def _trigger_render(self, current_time):
        """
        Mark that a render is needed.
        
        Args:
            current_time: Current pygame tick time
        """
        self.last_action_time = current_time
        self.pending_render = True
    
    def _apply_menu_settings(self, current_time):
        """
        Apply changed settings from the menu to the renderer.
        
        Args:
            current_time: Current pygame tick time
        """
        custom_formula = self.menu.custom_formula if self.menu.custom_formula else ''
        
        changed = self.renderer.update_settings(
            max_iter=self.menu.max_iter,
            colormap=self.menu.get_colormap(),
            func_id=self.menu.func_id,
            escape_radius=self.menu.escape_radius,
            custom_formula=custom_formula,
            julia_mode=self.menu.julia_mode,
            julia_c_real=self.menu.julia_c_real,
            julia_c_imag=self.menu.julia_c_imag
        )

        formula_error = self.renderer.get_last_formula_error()
        if formula_error:
            self.menu.formula_error = formula_error
            return
        self.menu.formula_error = None

        if not changed:
            return
        
        # Clear history since settings changed
        self.display.clear_history()
        self._trigger_render(current_time)
    
    def _handle_gpu_toggle(self, current_time):
        """
        Toggle GPU acceleration on/off.
        
        Args:
            current_time: Current pygame tick time
        """
        new_gpu_state = self.renderer.toggle_gpu()
        self._update_menu_gpu_status()
        
        # Update title with new status
        gpu_status = "GPU" if new_gpu_state else "CPU"
        self.display.set_title(
            f"Mandelbrot Set [{gpu_status}] - Scroll to zoom, drag to pan, R to reset"
        )
        
        # Clear cache and trigger re-render
        self.display.clear_history()
        self._trigger_render(current_time)
    
    def _update_menu_gpu_status(self):
        """Update the menu with current GPU status."""
        gpu_info = self.renderer.get_gpu_info()
        self.menu.update_gpu_status(
            gpu_info['available'],
            gpu_info['enabled'],
            gpu_info['device']
        )
    
    def _check_render_result(self):
        """Check for completed async render and update display."""
        result, actual_bounds = self.renderer.get_result()
        if result is not None:
            self.display.update_surface(result, actual_bounds)
            self.pending_render = False
            self.display.set_title(
                "Mandelbrot Set - Scroll to zoom, drag to pan, R to reset"
            )
    
    def _maybe_start_render(self, current_time):
        """
        Start a new async render if conditions are met.
        
        Uses debouncing to avoid starting renders while user is
        actively panning/zooming.
        
        Args:
            current_time: Current pygame tick time
        """
        if self.pending_render:
            if current_time - self.last_action_time > self.RENDER_DELAY_MS:
                self.renderer.compute_async(
                    self.x_min, self.x_max, 
                    self.y_min, self.y_max
                )
                self.display.set_title("Computing...")
    
    def _save_high_res_image(self):
        """Export a high-resolution image of the current view."""
        from compute import (
            compute_mandelbrot, compute_mandelbrot_custom,
            apply_colormap_smooth, downscale_2x
        )
        
        compute_funcs = {
            'compute': compute_mandelbrot,
            'compute_custom': compute_mandelbrot_custom,
            'colormap': apply_colormap_smooth,
            'downscale': downscale_2x
        }
        
        self.exporter.export(
            self.bounds, 
            self.renderer, 
            compute_funcs,
            self.display
        )
    
    def _cleanup(self):
        """
        Clean up all resources.
        
        Called on application exit to prevent resource leaks and
        ensure graceful shutdown.
        """
        if self._cleaned_up:
            return
        self._cleaned_up = True
        
        # Stop renderer threads
        if self.renderer:
            self.renderer.cleanup()
        
        # Clean up display
        if self.display:
            self.display.cleanup()
