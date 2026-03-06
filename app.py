"""Main application module for the Mandelbrot visualizer."""
import atexit, os
from datetime import datetime
import pygame, numpy as np
from .renderer import MandelbrotRenderer
from .menu import Menu
from .compute import compute_mandelbrot, compute_mandelbrot_custom, apply_colormap_smooth, downscale_2x, warmup_jit


class MandelbrotApp:
    """Main application class handling pygame window, events, and rendering."""
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_MAX_ITER = 800, 800, 500
    MAX_HISTORY, RENDER_DELAY_MS = 5, 25
    DEFAULT_BOUNDS = (-2.5, 1.0, -1.75, 1.75)
    ZOOM_IN_FACTOR, ZOOM_OUT_FACTOR = 0.85, 1.18

    def __init__(self, width=None, height=None, max_iter=None):
        self.width = width or self.DEFAULT_WIDTH
        self.height = height or self.DEFAULT_HEIGHT
        self.max_iter = max_iter or self.DEFAULT_MAX_ITER
        self.x_min, self.x_max, self.y_min, self.y_max = self.DEFAULT_BOUNDS
        self.screen = self.clock = self.renderer = self.menu = None
        self.current_surface = self.current_rgb = self.render_bounds = None
        self.render_history = []
        self.dragging, self.drag_start, self.drag_start_bounds = False, None, None
        self.last_action_time, self.pending_render = 0, False
        self.running, self._cleanup_registered, self._cleaned_up = False, False, False

    def run(self):
        self._init_pygame()
        self._init_components()
        self._warmup_and_initial_render()
        if not self._cleanup_registered:
            atexit.register(self._cleanup)
            self._cleanup_registered = True
        self.running = True
        while self.running:
            current_time = pygame.time.get_ticks()
            self._handle_events(current_time)
            if self.menu.save_requested:
                self.menu.save_requested = False
                self._save_high_res_image()
            if self.menu.gpu_toggle_requested:
                self.menu.gpu_toggle_requested = False
                self._handle_gpu_toggle(current_time)
            self._check_render_result()
            self._maybe_start_render(current_time)
            self._draw()
            self.clock.tick(60)
        self._cleanup()

    def _cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        if self.renderer:
            self.renderer.cleanup()
        self.current_surface = self.current_rgb = None
        self.render_history.clear()
        try:
            pygame.quit()
        except Exception:
            pass

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
        pygame.display.set_caption("Mandelbrot Set - Scroll to zoom, drag to pan")
        self.clock = pygame.time.Clock()

    def _init_components(self):
        self.renderer = MandelbrotRenderer(self.width, self.height, self.max_iter, use_gpu=None)
        self.menu = Menu(self.width - 250, 10)
        self.menu.max_iter = self.max_iter
        self._update_menu_gpu_status()

    def _update_menu_gpu_status(self):
        gpu_info = self.renderer.get_gpu_info()
        self.menu.update_gpu_status(gpu_info['available'], gpu_info['enabled'], gpu_info['device'])

    def _warmup_and_initial_render(self):
        pygame.display.set_caption("Compiling (first run only)...")
        warmup_jit(self.renderer.colormap)
        if self.renderer.use_gpu and self.renderer._gpu_compute:
            pygame.display.set_caption("Warming up GPU...")
            self.renderer._gpu_compute.warmup(self.renderer.colormap)
        w, h = self.x_max - self.x_min, self.y_max - self.y_min
        mx, my = w * self.renderer.margin, h * self.renderer.margin
        init_bounds = (self.x_min - mx, self.x_max + mx, self.y_min - my, self.y_max + my)
        if self.renderer.use_gpu and self.renderer._gpu_compute:
            data = self.renderer._gpu_compute.compute_mandelbrot(
                init_bounds[0], init_bounds[1], init_bounds[2], init_bounds[3],
                self.renderer.render_width, self.renderer.render_height, self.max_iter)
        else:
            data = compute_mandelbrot(init_bounds[0], init_bounds[1], init_bounds[2], init_bounds[3],
                self.renderer.render_width, self.renderer.render_height, self.max_iter)
        apply_colormap_smooth(data, self.max_iter, self.renderer.colormap, self.renderer.rgb_hi)
        downscale_2x(self.renderer.rgb_hi, self.renderer.rgb)
        self.current_rgb = np.flipud(self.renderer.rgb).copy()
        self.renderer.data_cache[:] = data
        self.renderer.cache_bounds = self.renderer.actual_bounds = init_bounds
        self.renderer._start_prefetch(init_bounds)
        self.current_surface = pygame.surfarray.make_surface(self.current_rgb.swapaxes(0, 1))
        self.screen.blit(self.current_surface, (0, 0))
        pygame.display.flip()
        self.render_bounds = init_bounds
        self.render_history = [(self.current_surface.copy(), init_bounds)]
        gpu_status = "GPU" if self.renderer.get_gpu_info()['enabled'] else "CPU"
        pygame.display.set_caption(f"Mandelbrot Set [{gpu_status}] - Scroll to zoom, drag to pan, R to reset")

    def _handle_events(self, current_time):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            menu_handled, need_recompute = self.menu.handle_event(event)
            if need_recompute:
                self._apply_menu_settings(current_time)
            if menu_handled:
                continue
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
        custom_formula = self.menu.custom_formula if self.menu.custom_formula else ''
        self.renderer.update_settings(max_iter=self.menu.max_iter, colormap=self.menu.get_colormap(),
            func_id=self.menu.func_id, escape_radius=self.menu.escape_radius, custom_formula=custom_formula,
            julia_mode=self.menu.julia_mode, julia_c_real=self.menu.julia_c_real, julia_c_imag=self.menu.julia_c_imag)
        self.last_action_time, self.pending_render = current_time, True
        self.render_history.clear()

    def _handle_gpu_toggle(self, current_time):
        new_gpu_state = self.renderer.toggle_gpu()
        self._update_menu_gpu_status()
        pygame.display.set_caption(f"Mandelbrot Set [{'GPU' if new_gpu_state else 'CPU'}] - Scroll to zoom, drag to pan, R to reset")
        self.last_action_time, self.pending_render = current_time, True
        self.render_history.clear()

    def _handle_zoom(self, event, current_time):
        if self.menu.point_in_menu(pygame.mouse.get_pos()):
            return
        mx, my = pygame.mouse.get_pos()
        cx = self.x_min + (self.x_max - self.x_min) * mx / self.width
        cy = self.y_min + (self.y_max - self.y_min) * (self.height - my) / self.height
        zoom = self.ZOOM_IN_FACTOR if event.y > 0 else self.ZOOM_OUT_FACTOR
        new_w, new_h = (self.x_max - self.x_min) * zoom, (self.y_max - self.y_min) * zoom
        x_ratio = (cx - self.x_min) / (self.x_max - self.x_min)
        y_ratio = (cy - self.y_min) / (self.y_max - self.y_min)
        self.x_min, self.x_max = cx - x_ratio * new_w, cx + (1 - x_ratio) * new_w
        self.y_min, self.y_max = cy - y_ratio * new_h, cy + (1 - y_ratio) * new_h
        self.last_action_time, self.pending_render = current_time, True

    def _handle_mouse_down(self, event):
        if event.button == 1 and not self.menu.point_in_menu(event.pos):
            self.dragging, self.drag_start = True, pygame.mouse.get_pos()
            self.drag_start_bounds = (self.x_min, self.x_max, self.y_min, self.y_max)

    def _handle_mouse_up(self, event, current_time):
        if event.button == 1:
            self.dragging = False
            self.last_action_time, self.pending_render = current_time, True

    def _handle_mouse_motion(self, event):
        if self.dragging and self.drag_start:
            mx, my = pygame.mouse.get_pos()
            dx = (self.drag_start[0] - mx) / self.width * (self.drag_start_bounds[1] - self.drag_start_bounds[0])
            dy = (my - self.drag_start[1]) / self.height * (self.drag_start_bounds[3] - self.drag_start_bounds[2])
            self.x_min, self.x_max = self.drag_start_bounds[0] + dx, self.drag_start_bounds[1] + dx
            self.y_min, self.y_max = self.drag_start_bounds[2] + dy, self.drag_start_bounds[3] + dy

    def _handle_key(self, event, current_time):
        if event.key == pygame.K_r:
            self.x_min, self.x_max, self.y_min, self.y_max = self.DEFAULT_BOUNDS
            self.last_action_time, self.pending_render = current_time, True
            self.render_history.clear()
        elif event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_META:
            self._save_high_res_image()

    def _save_high_res_image(self):
        scale, render_scale = 4, 2
        hi_width, hi_height = self.width * scale, self.height * scale
        rw, rh = hi_width * render_scale, hi_height * render_scale
        pygame.display.set_caption("Saving high-res image... (this may take a moment)")
        pygame.display.flip()
        if self.renderer._prepared_formula is not None:
            data = compute_mandelbrot_custom(self.x_min, self.x_max, self.y_min, self.y_max, rw, rh,
                self.renderer.max_iter, self.renderer._prepared_formula, self.renderer.escape_radius,
                self.renderer.julia_mode, self.renderer.julia_c_real,
                self.renderer.julia_c_imag)
        elif self.renderer.use_gpu and self.renderer._gpu_compute:
            data = self.renderer._gpu_compute.compute_mandelbrot(self.x_min, self.x_max, self.y_min, self.y_max,
                rw, rh, self.renderer.max_iter, self.renderer.func_id, self.renderer.escape_radius,
                self.renderer.julia_mode, self.renderer.julia_c_real, self.renderer.julia_c_imag)
        else:
            data = compute_mandelbrot(self.x_min, self.x_max, self.y_min, self.y_max, rw, rh,
                self.renderer.max_iter, func_id=self.renderer.func_id, escape_radius=self.renderer.escape_radius,
                julia_mode=self.renderer.julia_mode, julia_c_real=self.renderer.julia_c_real,
                julia_c_imag=self.renderer.julia_c_imag)
        hi_rgb = np.empty((rh, rw, 3), dtype=np.uint8)
        if self.renderer.use_gpu and self.renderer._gpu_compute:
            self.renderer._gpu_compute.apply_colormap_smooth(data, self.renderer.max_iter, self.renderer.colormap, hi_rgb)
        else:
            apply_colormap_smooth(data, self.renderer.max_iter, self.renderer.colormap, hi_rgb)
        final_rgb = np.empty((hi_height, hi_width, 3), dtype=np.uint8)
        if self.renderer.use_gpu and self.renderer._gpu_compute:
            self.renderer._gpu_compute.downscale_2x(hi_rgb, final_rgb)
        else:
            downscale_2x(hi_rgb, final_rgb)
        final_rgb = np.flipud(final_rgb)
        surface = pygame.surfarray.make_surface(final_rgb.swapaxes(0, 1))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(os.path.expanduser("~/Desktop"), f"mandelbrot_{timestamp}.png")
        pygame.image.save(surface, filename)
        pygame.display.set_caption(f"Saved: {os.path.basename(filename)} - Mandelbrot Set")
        print(f"High-resolution image saved to: {filename}")

    def _check_render_result(self):
        result, actual_bounds = self.renderer.get_result()
        if result is not None:
            self.current_rgb = result
            self.current_surface = pygame.surfarray.make_surface(self.current_rgb.swapaxes(0, 1))
            self.render_bounds, self.pending_render = actual_bounds, False
            pygame.display.set_caption("Mandelbrot Set - Scroll to zoom, drag to pan, R to reset")
            self.render_history.append((self.current_surface.copy(), actual_bounds))
            if len(self.render_history) > self.MAX_HISTORY:
                self.render_history.pop(0)

    def _maybe_start_render(self, current_time):
        if self.pending_render and current_time - self.last_action_time > self.RENDER_DELAY_MS:
            self.renderer.compute_async(self.x_min, self.x_max, self.y_min, self.y_max)
            pygame.display.set_caption("Computing...")

    def _draw(self):
        self.screen.fill((0, 0, 0))
        for hist_surface, hist_bounds in self.render_history:
            self._blit_surface_to_view(hist_surface, hist_bounds)
        self._blit_surface_to_view(self.current_surface, self.render_bounds)
        self.menu.draw(self.screen)
        pygame.display.flip()

    def _blit_surface_to_view(self, surface, bounds):
        """Transform and blit a rendered surface to match current view."""
        surf_w, surf_h = surface.get_width(), surface.get_height()
        bx_min, bx_max, by_min, by_max = bounds
        bound_w, bound_h = bx_max - bx_min, by_max - by_min
        src_left = (self.x_min - bx_min) / bound_w * surf_w
        src_right = (self.x_max - bx_min) / bound_w * surf_w
        src_bottom = (self.y_min - by_min) / bound_h * surf_h
        src_top = (self.y_max - by_min) / bound_h * surf_h
        src_left_c, src_right_c = max(0, min(surf_w, src_left)), max(0, min(surf_w, src_right))
        src_bottom_c, src_top_c = max(0, min(surf_h, src_bottom)), max(0, min(surf_h, src_top))
        src_w, src_h = src_right_c - src_left_c, src_top_c - src_bottom_c
        if src_w <= 0 or src_h <= 0:
            return
        view_w, view_h = src_right - src_left, src_top - src_bottom
        if view_w <= 0 or view_h <= 0:
            return
        dst_left = (src_left_c - src_left) / view_w * self.width
        dst_right = self.width - (src_right - src_right_c) / view_w * self.width
        dst_bottom = (src_bottom_c - src_bottom) / view_h * self.height
        dst_top = self.height - (src_top - src_top_c) / view_h * self.height
        dst_w, dst_h = dst_right - dst_left, dst_top - dst_bottom
        if dst_w <= 0 or dst_h <= 0:
            return
        try:
            src_rect = pygame.Rect(int(src_left_c), int(surf_h - src_top_c), int(src_w), int(src_h))
            subsurface = surface.subsurface(src_rect)
            scaled = pygame.transform.smoothscale(subsurface, (int(dst_w), int(dst_h)))
            self.screen.blit(scaled, (int(dst_left), int(self.height - dst_top)))
        except ValueError:
            pass


def run(width=None, height=None, max_iter=None):
    """Run the Mandelbrot visualizer."""
    app = MandelbrotApp(width, height, max_iter)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        app._cleanup()