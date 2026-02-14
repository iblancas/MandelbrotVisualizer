"""
Main application module for the Mandelbrot visualizer.

Contains the MandelbrotApp class which handles:
- Window setup and main loop
- User input (zoom, pan, keyboard)
- Rendering and display
- Interaction between renderer and menu
"""

import pygame
import numpy as np
from .renderer import MandelbrotRenderer
from .menu import Menu
from .colormaps import COLORMAPS
from .compute import (
    compute_mandelbrot,
    compute_mandelbrot_custom,
    apply_colormap_smooth,
    downscale_2x,
    warmup_jit
)


class MandelbrotApp:
    """
    Main application class for the Mandelbrot visualizer.
    
    Handles the pygame window, event loop, and coordinates
    between the renderer, menu, and display.
    """
    
    # Default configuration
    DEFAULT_WIDTH = 800
    DEFAULT_HEIGHT = 800
    DEFAULT_MAX_ITER = 500
    MAX_HISTORY = 5  # Number of previous renders to keep for zoom-out
    RENDER_DELAY_MS = 25  # Delay before starting render after user action
    
    # Default view bounds (shows classic Mandelbrot overview)
    DEFAULT_BOUNDS = (-2.5, 1.0, -1.75, 1.75)  # x_min, x_max, y_min, y_max
    
    # Zoom factors
    ZOOM_IN_FACTOR = 0.85
    ZOOM_OUT_FACTOR = 1.18
    
    def __init__(self, width=None, height=None, max_iter=None):
        """
        Initialize the application.
        
        Args:
            width: Window width in pixels (default 800)
            height: Window height in pixels (default 800)
            max_iter: Maximum iteration count (default 500)
        """
        self.width = width or self.DEFAULT_WIDTH
        self.height = height or self.DEFAULT_HEIGHT
        self.max_iter = max_iter or self.DEFAULT_MAX_ITER
        
        # View bounds in complex plane
        self.x_min, self.x_max, self.y_min, self.y_max = self.DEFAULT_BOUNDS
        
        # Pygame state (initialized in run())
        self.screen = None
        self.clock = None
        
        # Components
        self.renderer = None
        self.menu = None
        
        # Display state
        self.current_surface = None
        self.current_rgb = None
        self.render_bounds = None
        self.render_history = []
        
        # Input state
        self.dragging = False
        self.drag_start = None
        self.drag_start_bounds = None
        
        # Render timing
        self.last_action_time = 0
        self.pending_render = False
        
        self.running = False
    
    def run(self):
        """Run the application main loop."""
        self._init_pygame()
        self._init_components()
        self._warmup_and_initial_render()
        
        self.running = True
        while self.running:
            current_time = pygame.time.get_ticks()
            
            self._handle_events(current_time)
            
            # Check for save request from menu
            if self.menu.save_requested:
                self.menu.save_requested = False
                self._save_high_res_image()
            
            self._check_render_result()
            self._maybe_start_render(current_time)
            self._draw()
            
            self.clock.tick(60)
        
        pygame.quit()
    
    def _init_pygame(self):
        """Initialize pygame and create window."""
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Mandelbrot Set - Scroll to zoom, drag to pan")
        self.clock = pygame.time.Clock()
    
    def _init_components(self):
        """Initialize renderer and menu."""
        self.renderer = MandelbrotRenderer(self.width, self.height, self.max_iter)
        
        # Menu in top-right corner
        self.menu = Menu(self.width - 250, 10)
        self.menu.max_iter = self.max_iter
    
    def _warmup_and_initial_render(self):
        """Warm up JIT and do initial render."""
        pygame.display.set_caption("Compiling (first run only)...")
        warmup_jit(self.renderer.colormap)
        
        # Calculate bounds with margin
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min
        mx = w * self.renderer.margin
        my = h * self.renderer.margin
        init_bounds = (
            self.x_min - mx, self.x_max + mx,
            self.y_min - my, self.y_max + my
        )
        
        # Compute initial image
        data = compute_mandelbrot(
            init_bounds[0], init_bounds[1], init_bounds[2], init_bounds[3],
            self.renderer.render_width, self.renderer.render_height,
            self.max_iter
        )
        apply_colormap_smooth(
            data, self.max_iter, self.renderer.colormap, self.renderer.rgb_hi
        )
        downscale_2x(self.renderer.rgb_hi, self.renderer.rgb)
        self.current_rgb = np.flipud(self.renderer.rgb).copy()
        
        # Initialize renderer cache
        self.renderer.data_cache[:] = data
        self.renderer.cache_bounds = init_bounds
        self.renderer.actual_bounds = init_bounds
        self.renderer._start_prefetch(init_bounds)
        
        # Create display surface
        self.current_surface = pygame.surfarray.make_surface(
            self.current_rgb.swapaxes(0, 1)
        )
        self.screen.blit(self.current_surface, (0, 0))
        pygame.display.flip()
        
        self.render_bounds = init_bounds
        self.render_history = [(self.current_surface.copy(), init_bounds)]
        
        pygame.display.set_caption(
            "Mandelbrot Set - Scroll to zoom, drag to pan, R to reset"
        )
    
    def _handle_events(self, current_time):
        """Process all pending pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            
            # Menu gets first crack at events
            menu_handled, need_recompute = self.menu.handle_event(event)
            if need_recompute:
                self._apply_menu_settings(current_time)
            if menu_handled:
                continue
            
            # Handle other events
            if event.type == pygame.MOUSEWHEEL:
                self._handle_zoom(event, current_time)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._handle_mouse_up(event, current_time)
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(event)
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event, current_time)
    
    def _apply_menu_settings(self, current_time):
        """Apply changed settings from menu."""
        new_colormap = self.menu.get_colormap()
        # Pass custom_formula: None to keep current, '' to clear, or formula string
        custom_formula = self.menu.custom_formula if self.menu.custom_formula else ''
        self.renderer.update_settings(
            max_iter=self.menu.max_iter,
            colormap=new_colormap,
            func_id=self.menu.func_id,
            escape_radius=self.menu.escape_radius,
            custom_formula=custom_formula
        )
        self.last_action_time = current_time
        self.pending_render = True
        self.render_history.clear()
    
    def _handle_zoom(self, event, current_time):
        """Handle mouse wheel zoom."""
        if self.menu.point_in_menu(pygame.mouse.get_pos()):
            return
        
        mx, my = pygame.mouse.get_pos()
        
        # Convert mouse position to complex plane coordinates
        cx = self.x_min + (self.x_max - self.x_min) * mx / self.width
        cy = self.y_min + (self.y_max - self.y_min) * (self.height - my) / self.height
        
        # Zoom factor (scroll up = zoom in)
        zoom = self.ZOOM_IN_FACTOR if event.y > 0 else self.ZOOM_OUT_FACTOR
        
        # Calculate new bounds centered on mouse position
        new_width = (self.x_max - self.x_min) * zoom
        new_height = (self.y_max - self.y_min) * zoom
        
        x_ratio = (cx - self.x_min) / (self.x_max - self.x_min)
        y_ratio = (cy - self.y_min) / (self.y_max - self.y_min)
        
        self.x_min = cx - x_ratio * new_width
        self.x_max = cx + (1 - x_ratio) * new_width
        self.y_min = cy - y_ratio * new_height
        self.y_max = cy + (1 - y_ratio) * new_height
        
        self.last_action_time = current_time
        self.pending_render = True
    
    def _handle_mouse_down(self, event):
        """Handle mouse button press."""
        if event.button == 1:  # Left click
            if not self.menu.point_in_menu(event.pos):
                self.dragging = True
                self.drag_start = pygame.mouse.get_pos()
                self.drag_start_bounds = (
                    self.x_min, self.x_max, self.y_min, self.y_max
                )
    
    def _handle_mouse_up(self, event, current_time):
        """Handle mouse button release."""
        if event.button == 1:
            self.dragging = False
            self.last_action_time = current_time
            self.pending_render = True
    
    def _handle_mouse_motion(self, event):
        """Handle mouse movement (for dragging)."""
        if self.dragging and self.drag_start:
            mx, my = pygame.mouse.get_pos()
            
            # Calculate drag distance in complex plane units
            dx = ((self.drag_start[0] - mx) / self.width * 
                  (self.drag_start_bounds[1] - self.drag_start_bounds[0]))
            dy = ((my - self.drag_start[1]) / self.height * 
                  (self.drag_start_bounds[3] - self.drag_start_bounds[2]))
            
            self.x_min = self.drag_start_bounds[0] + dx
            self.x_max = self.drag_start_bounds[1] + dx
            self.y_min = self.drag_start_bounds[2] + dy
            self.y_max = self.drag_start_bounds[3] + dy
    
    def _handle_key(self, event, current_time):
        """Handle keyboard input."""
        if event.key == pygame.K_r:
            # Reset to default view
            self.x_min, self.x_max, self.y_min, self.y_max = self.DEFAULT_BOUNDS
            self.last_action_time = current_time
            self.pending_render = True
            self.render_history.clear()
        elif event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_META:
            # Cmd+S to save high-res image
            self._save_high_res_image()
    
    def _save_high_res_image(self):
        """Save a high-resolution image of the current view."""
        import os
        from datetime import datetime
        
        # Resolution multiplier (4x for high-res)
        scale = 4
        hi_width = self.width * scale
        hi_height = self.height * scale
        
        # Create larger arrays for high-res render
        render_scale = 2  # 2x supersampling
        rw = hi_width * render_scale
        rh = hi_height * render_scale
        
        pygame.display.set_caption("Saving high-res image... (this may take a moment)")
        pygame.display.flip()
        
        # Compute at high resolution
        if self.renderer._prepared_formula is not None:
            # Use custom formula (slower)
            data = compute_mandelbrot_custom(
                self.x_min, self.x_max, self.y_min, self.y_max,
                rw, rh,
                self.renderer.max_iter,
                self.renderer._prepared_formula,
                self.renderer.escape_radius
            )
        else:
            # Use JIT-compiled function
            data = compute_mandelbrot(
                self.x_min, self.x_max, self.y_min, self.y_max,
                rw, rh,
                self.renderer.max_iter,
                func_id=self.renderer.func_id,
                escape_radius=self.renderer.escape_radius
            )
        
        # Apply colormap
        hi_rgb = np.empty((rh, rw, 3), dtype=np.uint8)
        apply_colormap_smooth(
            data, self.renderer.max_iter, self.renderer.colormap, hi_rgb
        )
        
        # Downscale with 2x supersampling
        final_rgb = np.empty((hi_height, hi_width, 3), dtype=np.uint8)
        downscale_2x(hi_rgb, final_rgb)
        
        # Flip for correct orientation
        final_rgb = np.flipud(final_rgb)
        
        # Create surface and save
        surface = pygame.surfarray.make_surface(final_rgb.swapaxes(0, 1))
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        desktop_path = os.path.expanduser("~/Desktop")
        filename = os.path.join(desktop_path, f"mandelbrot_{timestamp}.png")
        
        pygame.image.save(surface, filename)
        pygame.display.set_caption(f"Saved: {os.path.basename(filename)} - Mandelbrot Set")
        
        # Reset title after a moment (will be reset on next action anyway)
        print(f"High-resolution image saved to: {filename}")

    def _check_render_result(self):
        """Check if async render has completed."""
        result, actual_bounds = self.renderer.get_result()
        if result is not None:
            self.current_rgb = result
            self.current_surface = pygame.surfarray.make_surface(
                self.current_rgb.swapaxes(0, 1)
            )
            self.render_bounds = actual_bounds
            self.pending_render = False
            pygame.display.set_caption(
                "Mandelbrot Set - Scroll to zoom, drag to pan, R to reset"
            )
            
            # Add to history
            self.render_history.append((self.current_surface.copy(), actual_bounds))
            if len(self.render_history) > self.MAX_HISTORY:
                self.render_history.pop(0)
    
    def _maybe_start_render(self, current_time):
        """Start a new render if conditions are met."""
        if self.pending_render and current_time - self.last_action_time > self.RENDER_DELAY_MS:
            self.renderer.compute_async(
                self.x_min, self.x_max, self.y_min, self.y_max
            )
            pygame.display.set_caption("Computing...")
    
    def _draw(self):
        """Draw the current frame."""
        self.screen.fill((0, 0, 0))
        
        # Draw from history (oldest first, so newer ones overdraw)
        for hist_surface, hist_bounds in self.render_history:
            self._blit_surface_to_view(hist_surface, hist_bounds)
        
        # Draw current surface on top
        self._blit_surface_to_view(self.current_surface, self.render_bounds)
        
        # Draw menu overlay
        self.menu.draw(self.screen)
        
        pygame.display.flip()
    
    def _blit_surface_to_view(self, surface, bounds):
        """
        Blit a rendered surface to the screen, transforming for current view.
        
        This handles the case where the rendered bounds don't exactly match
        the current view (e.g., during panning/zooming).
        """
        surf_w = surface.get_width()
        surf_h = surface.get_height()
        
        bx_min, bx_max, by_min, by_max = bounds
        bound_w = bx_max - bx_min
        bound_h = by_max - by_min
        
        # Map current view coordinates to pixel positions in this surface
        src_left = (self.x_min - bx_min) / bound_w * surf_w
        src_right = (self.x_max - bx_min) / bound_w * surf_w
        src_bottom = (self.y_min - by_min) / bound_h * surf_h
        src_top = (self.y_max - by_min) / bound_h * surf_h
        
        # Clamp to valid image bounds
        src_left_clamped = max(0, min(surf_w, src_left))
        src_right_clamped = max(0, min(surf_w, src_right))
        src_bottom_clamped = max(0, min(surf_h, src_bottom))
        src_top_clamped = max(0, min(surf_h, src_top))
        
        src_w = src_right_clamped - src_left_clamped
        src_h = src_top_clamped - src_bottom_clamped
        
        if src_w <= 0 or src_h <= 0:
            return
        
        view_w = src_right - src_left
        view_h = src_top - src_bottom
        
        if view_w <= 0 or view_h <= 0:
            return
        
        # Calculate destination rectangle
        dst_left = (src_left_clamped - src_left) / view_w * self.width
        dst_right = self.width - (src_right - src_right_clamped) / view_w * self.width
        dst_bottom = (src_bottom_clamped - src_bottom) / view_h * self.height
        dst_top = self.height - (src_top - src_top_clamped) / view_h * self.height
        
        dst_w = dst_right - dst_left
        dst_h = dst_top - dst_bottom
        
        if dst_w <= 0 or dst_h <= 0:
            return
        
        try:
            src_rect = pygame.Rect(
                int(src_left_clamped),
                int(surf_h - src_top_clamped),
                int(src_w),
                int(src_h)
            )
            subsurface = surface.subsurface(src_rect)
            scaled = pygame.transform.smoothscale(
                subsurface, (int(dst_w), int(dst_h))
            )
            self.screen.blit(scaled, (int(dst_left), int(self.height - dst_top)))
        except ValueError:
            pass  # Subsurface out of bounds, skip this frame


def run(width=None, height=None, max_iter=None):
    """
    Run the Mandelbrot visualizer.
    
    Args:
        width: Window width (default 800)
        height: Window height (default 800)
        max_iter: Maximum iterations (default 500)
    """
    app = MandelbrotApp(width, height, max_iter)
    try:
        app.run()
    except KeyboardInterrupt:
        quit()