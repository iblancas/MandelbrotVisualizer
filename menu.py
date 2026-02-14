"""Settings menu for the Mandelbrot visualizer with dropdowns and gradient editor."""

import colorsys
import json
import os
import re
import numpy as np
import pygame
from .colormaps import COLORMAPS, NUM_COLORS


def _load_settings():
    path = os.path.join(os.path.dirname(__file__), 'settings.json')
    with open(path) as f:
        return json.load(f)

_SETTINGS = _load_settings()


def _hsv_to_rgb(h, s, v):
    """HSV (0-1) to RGB (0-255)."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _rgb_to_hsv(r, g, b):
    """RGB (0-255) to HSV (0-1)."""
    return colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)


class TextInput:
    """Text input field component."""
    
    def __init__(self, x, y, width, height=24, placeholder=""):
        self.x, self.y, self.width, self.height = x, y, width, height
        self.text = ""
        self.placeholder = placeholder
        self.active = False
        self.cursor_pos = 0
        self._cursor_timer = 0
    
    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def handle_event(self, event):
        """Returns (handled, text_changed)."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            was_active = self.active
            self.active = self.rect.collidepoint(event.pos)
            return self.active or was_active, False
        
        if event.type == pygame.KEYDOWN and self.active:
            old = self.text
            key = event.key
            if key == pygame.K_BACKSPACE and self.cursor_pos > 0:
                self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                self.cursor_pos -= 1
            elif key == pygame.K_DELETE and self.cursor_pos < len(self.text):
                self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]
            elif key == pygame.K_LEFT:
                self.cursor_pos = max(0, self.cursor_pos - 1)
            elif key == pygame.K_RIGHT:
                self.cursor_pos = min(len(self.text), self.cursor_pos + 1)
            elif key in (pygame.K_HOME, pygame.K_END):
                self.cursor_pos = 0 if key == pygame.K_HOME else len(self.text)
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.active = False
                return True, old != self.text
            elif event.unicode and event.unicode.isprintable():
                self.text = self.text[:self.cursor_pos] + event.unicode + self.text[self.cursor_pos:]
                self.cursor_pos += 1
            return True, old != self.text
        return False, False
    
    def draw(self, screen, font):
        rect = self.rect
        pygame.draw.rect(screen, (60, 60, 70) if self.active else (50, 50, 55), rect)
        pygame.draw.rect(screen, (100, 140, 180) if self.active else (80, 80, 80), rect, 2 if self.active else 1)
        
        text = self.text or self.placeholder
        color = (220, 220, 220) if self.text else (120, 120, 120)
        surf = font.render(text, True, color)
        screen.blit(surf, (rect.left + 6, rect.centery - surf.get_height() // 2))
        
        if self.active:
            self._cursor_timer = (self._cursor_timer + 1) % 60
            if self._cursor_timer < 30:
                cx = rect.left + 6 + (font.render(self.text[:self.cursor_pos], True, color).get_width() if self.text else 0)
                pygame.draw.line(screen, (220, 220, 220), (cx, rect.top + 4), (cx, rect.bottom - 4), 2)


class Dropdown:
    """Dropdown select component."""
    
    def __init__(self, x, y, width, options, selected_idx=0):
        self.x, self.y, self.width = x, y, width
        self.height = 24
        self.options = options
        self.selected_idx = selected_idx
        self.expanded = False
        self.hovered_idx = -1
    
    @property
    def value(self):
        return self.options[self.selected_idx]
    
    def set_value(self, value):
        if value in self.options:
            self.selected_idx = self.options.index(value)
    
    def handle_event(self, event):
        """Returns (handled, value_changed)."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if pygame.Rect(self.x, self.y, self.width, self.height).collidepoint(mx, my):
                self.expanded = not self.expanded
                return True, False
            if self.expanded:
                item_y = self.y + self.height
                for i in range(len(self.options)):
                    if pygame.Rect(self.x, item_y, self.width, 22).collidepoint(mx, my):
                        old = self.selected_idx
                        self.selected_idx = i
                        self.expanded = False
                        return True, old != i
                    item_y += 22
                self.expanded = False
                return True, False
        elif event.type == pygame.MOUSEMOTION and self.expanded:
            self.hovered_idx = -1
            item_y = self.y + self.height
            for i in range(len(self.options)):
                if pygame.Rect(self.x, item_y, self.width, 22).collidepoint(event.pos):
                    self.hovered_idx = i
                    break
                item_y += 22
        return False, False
    
    def draw(self, screen, font):
        pygame.draw.rect(screen, (55, 55, 55), (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (100, 100, 100), (self.x, self.y, self.width, self.height), 1)
        screen.blit(font.render(str(self.value), True, (220, 220, 220)), (self.x + 8, self.y + 5))
        screen.blit(font.render("▲" if self.expanded else "▼", True, (150, 150, 150)), (self.x + self.width - 18, self.y + 5))
        
        if self.expanded:
            item_y = self.y + self.height
            for i, opt in enumerate(self.options):
                bg = (70, 100, 70) if i == self.selected_idx else (65, 65, 65) if i == self.hovered_idx else (50, 50, 50)
                pygame.draw.rect(screen, bg, (self.x, item_y, self.width, 22))
                pygame.draw.rect(screen, (80, 80, 80), (self.x, item_y, self.width, 22), 1)
                color = (255, 255, 255) if i == self.selected_idx else (180, 180, 180)
                screen.blit(font.render(str(opt), True, color), (self.x + 8, item_y + 4))
                item_y += 22


class GradientEditor:
    """Popup gradient editor for custom color schemes."""
    
    def __init__(self, screen_width, screen_height):
        self.width, self.height = 500, 420
        self.x, self.y = (screen_width - self.width) // 2, (screen_height - self.height) // 2
        self.visible = False
        self.font = None
        
        self.stops = [(0.0, (0, 0, 0)), (0.5, (255, 100, 0)), (1.0, (255, 255, 255))]
        self.selected_stop = self.dragging_stop = self.editing_color = -1
        self.dragging_sv = self.dragging_hue = False
        self.dragging_slider = -1
        self.hsv = [0.0, 1.0, 1.0]  # Current H, S, V
        self.sv_surface = self.hue_surface = None
    
    def _init(self):
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 12)
    
    def _bar_rect(self):
        return self.x + 20, self.y + 60, self.width - 40, 30
    
    def _picker_rect(self):
        return self.x + 20, self.y + 130, 150, 150  # SV square
    
    def _create_hue_surface(self):
        self.hue_surface = pygame.Surface((20, 150))
        for y in range(150):
            pygame.draw.line(self.hue_surface, _hsv_to_rgb(y / 150, 1, 1), (0, y), (20, y))
    
    def _create_sv_surface(self):
        self.sv_surface = pygame.Surface((150, 150))
        for x in range(150):
            for y in range(150):
                self.sv_surface.set_at((x, y), _hsv_to_rgb(self.hsv[0], x / 150, 1 - y / 150))
    
    def show(self):
        self.visible = True
        self.selected_stop = self.dragging_stop = self.editing_color = -1
        self.dragging_sv = self.dragging_hue = False
        self.dragging_slider = -1
        if not self.hue_surface:
            self._create_hue_surface()
    
    def hide(self):
        self.visible = False
    
    def _sync_color(self):
        if 0 <= self.editing_color < len(self.stops):
            self.stops[self.editing_color] = (self.stops[self.editing_color][0], _hsv_to_rgb(*self.hsv))
    
    def _sync_hsv(self):
        if 0 <= self.editing_color < len(self.stops):
            self.hsv = list(_rgb_to_hsv(*self.stops[self.editing_color][1]))
            self._create_sv_surface()
    
    def get_colormap(self):
        colors = np.zeros((NUM_COLORS, 3), dtype=np.uint8)
        stops = sorted(self.stops, key=lambda s: s[0])
        if len(stops) == 0:
            return colors
        if len(stops) == 1:
            colors[:] = stops[0][1]
            return colors
        
        for i in range(NUM_COLORS):
            t = i / (NUM_COLORS - 1)
            left, right = stops[0], stops[-1]
            for j in range(len(stops) - 1):
                if stops[j][0] <= t <= stops[j + 1][0]:
                    left, right = stops[j], stops[j + 1]
                    break
            lt = 0 if right[0] == left[0] else (t - left[0]) / (right[0] - left[0])
            colors[i] = [int(left[1][c] * (1 - lt) + right[1][c] * lt) for c in range(3)]
        return colors
    
    def handle_event(self, event):
        """Returns (handled, should_apply)."""
        if not self.visible:
            return False, False
        
        bx, by, bw, bh = self._bar_rect()
        px, py, ps, _ = self._picker_rect()
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if not pygame.Rect(self.x, self.y, self.width, self.height).collidepoint(mx, my):
                self.hide()
                return True, False
            
            # Buttons
            if pygame.Rect(self.x + self.width - 80, self.y + self.height - 40, 60, 28).collidepoint(mx, my):
                self.hide()
                return True, True  # Apply
            if pygame.Rect(self.x + self.width - 150, self.y + self.height - 40, 60, 28).collidepoint(mx, my):
                self.hide()
                return True, False  # Cancel
            if pygame.Rect(self.x + 20, self.y + self.height - 40, 80, 28).collidepoint(mx, my):
                # Add stop at largest gap
                s = sorted(self.stops, key=lambda x: x[0])
                pos = 0.5
                if len(s) >= 2:
                    max_gap = 0
                    for i in range(len(s) - 1):
                        gap = s[i+1][0] - s[i][0]
                        if gap > max_gap:
                            max_gap, pos = gap, (s[i][0] + s[i+1][0]) / 2
                cmap = self.get_colormap()
                self.stops.append((pos, tuple(cmap[int(pos * (NUM_COLORS - 1))])))
                return True, False
            
            # Color picker (if editing)
            if self.editing_color >= 0:
                if pygame.Rect(px, py, ps, ps).collidepoint(mx, my):  # SV square
                    self.dragging_sv = True
                    self.hsv[1] = max(0, min(1, (mx - px) / ps))
                    self.hsv[2] = max(0, min(1, 1 - (my - py) / ps))
                    self._sync_color()
                    return True, False
                hue_x = px + ps + 10
                if pygame.Rect(hue_x, py, 20, 150).collidepoint(mx, my):  # Hue bar
                    self.dragging_hue = True
                    self.hsv[0] = max(0, min(1, (my - py) / 150))
                    self._create_sv_surface()
                    self._sync_color()
                    return True, False
                # RGB sliders
                slider_x, slider_y = self.x + 210, self.y + 160
                for i in range(3):
                    if pygame.Rect(slider_x, slider_y + i * 35, 180, 18).collidepoint(mx, my):
                        self.dragging_slider = i
                        self._update_slider(mx)
                        return True, False
            
            # Gradient stops
            for i, (p, _) in enumerate(self.stops):
                sx = bx + p * bw
                if pygame.Rect(sx - 8, by + bh, 16, 18).collidepoint(mx, my):
                    self.selected_stop = self.dragging_stop = self.editing_color = i
                    self._sync_hsv()
                    return True, False
            
            if pygame.Rect(bx, by, bw, bh).collidepoint(mx, my):
                self.selected_stop = self.editing_color = -1
            return True, False
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging_stop = -1
            self.dragging_sv = self.dragging_hue = False
            self.dragging_slider = -1
            return self.visible, False
        
        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if self.dragging_stop >= 0:
                self.stops[self.dragging_stop] = (max(0, min(1, (mx - bx) / bw)), self.stops[self.dragging_stop][1])
                return True, False
            if self.dragging_sv and self.editing_color >= 0:
                self.hsv[1] = max(0, min(1, (mx - px) / ps))
                self.hsv[2] = max(0, min(1, 1 - (my - py) / ps))
                self._sync_color()
                return True, False
            if self.dragging_hue and self.editing_color >= 0:
                self.hsv[0] = max(0, min(1, (my - py) / 150))
                self._create_sv_surface()
                self._sync_color()
                return True, False
            if self.dragging_slider >= 0 and self.editing_color >= 0:
                self._update_slider(mx)
                return True, False
        
        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
            if self.selected_stop >= 0 and len(self.stops) > 2:
                del self.stops[self.selected_stop]
                self.selected_stop = self.editing_color = -1
            return True, False
        
        return self.visible, False
    
    def _update_slider(self, mx):
        slider_x = self.x + 210
        val = max(0, min(255, int((mx - slider_x) / 180 * 255)))
        color = list(self.stops[self.editing_color][1])
        color[self.dragging_slider] = val
        self.stops[self.editing_color] = (self.stops[self.editing_color][0], tuple(color))
        self._sync_hsv()
    
    def draw(self, screen):
        if not self.visible:
            return
        self._init()
        
        # Background
        pygame.draw.rect(screen, (45, 45, 45), (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (100, 100, 100), (self.x, self.y, self.width, self.height), 2)
        
        screen.blit(self.font.render('Custom Gradient Editor', True, (220, 220, 220)), (self.x + 15, self.y + 10))
        screen.blit(self.font.render('Click stops to edit. Drag to move. Delete to remove.', True, (150, 150, 150)), (self.x + 15, self.y + 32))
        
        # Gradient bar
        bx, by, bw, bh = self._bar_rect()
        cmap = self.get_colormap()
        for px in range(bw):
            pygame.draw.line(screen, tuple(cmap[int(px / bw * (NUM_COLORS - 1))]), (bx + px, by), (bx + px, by + bh))
        pygame.draw.rect(screen, (100, 100, 100), (bx, by, bw, bh), 1)
        
        # Stops
        for i, (pos, color) in enumerate(self.stops):
            sx = int(bx + pos * bw)
            pts = [(sx, by + bh), (sx - 8, by + bh + 15), (sx + 8, by + bh + 15)]
            pygame.draw.polygon(screen, (255, 255, 100) if i == self.selected_stop else color, pts)
            pygame.draw.polygon(screen, (255, 255, 255) if i == self.selected_stop else (200, 200, 200), pts, 2 if i == self.selected_stop else 1)
        
        # Color picker
        if 0 <= self.editing_color < len(self.stops):
            self._draw_picker(screen)
        
        # Buttons
        for rect, text, bg in [
            ((self.x + 20, self.y + self.height - 40, 80, 28), 'Add Stop', (60, 80, 60)),
            ((self.x + self.width - 150, self.y + self.height - 40, 60, 28), 'Cancel', (80, 60, 60)),
            ((self.x + self.width - 80, self.y + self.height - 40, 60, 28), 'Apply', (60, 80, 100)),
        ]:
            pygame.draw.rect(screen, bg, rect)
            pygame.draw.rect(screen, tuple(c + 40 for c in bg), rect, 1)
            t = self.font.render(text, True, (220, 220, 220))
            screen.blit(t, (rect[0] + (rect[2] - t.get_width()) // 2, rect[1] + 6))
    
    def _draw_picker(self, screen):
        px, py, ps, _ = self._picker_rect()
        color = self.stops[self.editing_color][1]
        
        screen.blit(self.font.render(f'Position: {self.stops[self.editing_color][0]:.2f}', True, (180, 180, 180)), (px, py - 20))
        
        # SV square
        if self.sv_surface:
            screen.blit(self.sv_surface, (px, py))
        pygame.draw.rect(screen, (80, 80, 80), (px, py, ps, ps), 1)
        cx, cy = int(px + self.hsv[1] * ps), int(py + (1 - self.hsv[2]) * ps)
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 6, 2)
        pygame.draw.circle(screen, (0, 0, 0), (cx, cy), 5, 1)
        
        # Hue bar
        hx = px + ps + 10
        if self.hue_surface:
            screen.blit(self.hue_surface, (hx, py))
        pygame.draw.rect(screen, (80, 80, 80), (hx, py, 20, 150), 1)
        hy = int(py + self.hsv[0] * 150)
        pygame.draw.rect(screen, (255, 255, 255), (hx - 2, hy - 2, 24, 4))
        pygame.draw.rect(screen, (0, 0, 0), (hx - 2, hy - 2, 24, 4), 1)
        
        # Preview and RGB sliders
        slider_x, slider_y = self.x + 210, self.y + 160
        pygame.draw.rect(screen, color, (slider_x, py - 20, 50, 30))
        pygame.draw.rect(screen, (100, 100, 100), (slider_x, py - 20, 50, 30), 1)
        screen.blit(self.font.render(f'R:{color[0]} G:{color[1]} B:{color[2]}', True, (180, 180, 180)), (slider_x + 55, py - 12))
        
        for i, (lbl, val) in enumerate(zip('RGB', color)):
            sy = slider_y + i * 35
            screen.blit(self.font.render(f'{lbl}:', True, (180, 180, 180)), (slider_x - 20, sy + 2))
            for sx in range(180):
                pc = list(color)
                pc[i] = int(sx / 180 * 255)
                pygame.draw.line(screen, pc, (slider_x + sx, sy), (slider_x + sx, sy + 18))
            pygame.draw.rect(screen, (80, 80, 80), (slider_x, sy, 180, 18), 1)
            tx = int(slider_x + val / 255 * 180)
            pygame.draw.rect(screen, (220, 220, 220), (tx - 4, sy - 2, 8, 22))
            pygame.draw.rect(screen, (100, 100, 100), (tx - 4, sy - 2, 8, 22), 1)
            screen.blit(self.font.render(str(val), True, (220, 220, 220)), (slider_x + 190, sy + 2))


def parse_formula(formula):
    """Parse user formula and return (func_id, display_name) or (None, error)."""
    f = formula.lower().replace(' ', '').replace('·', '*').replace('×', '*')
    f = f.replace('^', '**').replace('²', '**2').replace('³', '**3')
    for i, sup in enumerate('⁴⁵⁶⁷⁸', 4):
        f = f.replace(sup, f'**{i}')
    f = f.replace('z_bar', 'conj(z)').replace('z̄', 'conj(z)')
    f = f.replace('e**z', 'exp(z)').replace('sin(z)+c', 'c*sin(z)').replace('cos(z)+c', 'c*cos(z)').replace('exp(z)+c', 'c*exp(z)')
    
    # Match against patterns from settings
    for p in _SETTINGS.get('formula_patterns', []):
        if f == p['pattern']:
            func_id = p['id']
            for fn in _SETTINGS['iteration_functions']:
                if fn['id'] == func_id:
                    return func_id, fn['name']
    
    # Try z^n + c
    match = re.match(r'^z\*\*(\d+)\+c$', f)
    if match:
        n = match.group(1)
        power_map = _SETTINGS.get('power_to_func_id', {})
        if n in power_map:
            return power_map[n], f'z^{n} + c'
        return None, f'z^{n} not supported (max 8)'
    
    return None, 'Unrecognized formula'


class Menu:
    """Settings menu with dropdowns and gradient editor."""
    
    def __init__(self, x, y, width=220, screen_width=800, screen_height=800):
        self.x, self.y, self.width = x, y, width
        self.expanded = False
        self.font = None
        
        defaults = _SETTINGS['default_settings']
        self.max_iter = defaults['max_iterations']
        self.colormap_name = defaults['colormap']
        self.func_id = defaults['function_id']
        self.escape_radius = defaults['escape_radius']
        
        self.funcs = [(f['name'], f['id']) for f in _SETTINGS['iteration_functions']]
        self.colormap_names = list(COLORMAPS.keys()) + ['Custom...']
        self.func_display = next((n for n, i in self.funcs if i == self.func_id), 'z^2 + c')
        self.custom_formula = None
        self.custom_colormap = None
        self.using_custom = False
        
        self.use_gpu = True
        self.gpu_available = False
        self.gpu_device_name = "Checking..."
        self.gpu_toggle_rect = None
        self.save_requested = False
        self.gpu_toggle_requested = False
        
        self.dropdowns = {}
        self.func_input = None
        self.formula_error = None
        self.save_button_rect = None
        self.gradient_editor = GradientEditor(screen_width, screen_height)
    
    def _init(self):
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 12)
    
    def _init_dropdowns(self):
        w = self.width - 16
        opts = _SETTINGS
        self.dropdowns = {
            'iter': Dropdown(self.x + 8, 0, w, [str(v) for v in opts['iteration_options']],
                           opts['iteration_options'].index(self.max_iter) if self.max_iter in opts['iteration_options'] else 0),
            'func': Dropdown(self.x + 8, 0, w, [n for n, _ in self.funcs],
                           next((i for i, (_, fid) in enumerate(self.funcs) if fid == self.func_id), 0)),
            'escape': Dropdown(self.x + 8, 0, w, [str(v) for v in opts['escape_radius_options']],
                              opts['escape_radius_options'].index(int(self.escape_radius)) if int(self.escape_radius) in opts['escape_radius_options'] else 0),
            'color': Dropdown(self.x + 8, 0, w, self.colormap_names,
                             self.colormap_names.index(self.colormap_name) if self.colormap_name in self.colormap_names else 0),
        }
        self.func_input = TextInput(self.x + 8, 0, w, 24, "Type formula: z^3+c")
    
    def get_rect(self):
        if not self.expanded:
            return pygame.Rect(self.x, self.y, 120, 24)
        h = 390 + (15 if self.formula_error else 0)
        for dd in self.dropdowns.values():
            if dd.expanded:
                h += len(dd.options) * 22
        return pygame.Rect(self.x, self.y, self.width, h)
    
    def get_colormap(self):
        return self.custom_colormap if self.using_custom and self.custom_colormap is not None else COLORMAPS[self.colormap_name]()
    
    def handle_event(self, event):
        """Returns (handled, need_recompute)."""
        if self.gradient_editor.visible:
            handled, apply = self.gradient_editor.handle_event(event)
            if apply:
                self.custom_colormap = self.gradient_editor.get_colormap()
                self.using_custom = True
                self.colormap_name = 'Custom...'
                if 'color' in self.dropdowns:
                    self.dropdowns['color'].set_value('Custom...')
                return True, True
            return handled, False
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self._handle_click(event.pos)
        elif event.type == pygame.KEYDOWN:
            return self._handle_key(event)
        elif event.type == pygame.MOUSEMOTION and self.expanded:
            for dd in self.dropdowns.values():
                dd.handle_event(event)
        return False, False
    
    def _handle_click(self, pos):
        mx, my = pos
        if pygame.Rect(self.x, self.y, 120, 24).collidepoint(mx, my):
            self.expanded = not self.expanded
            if self.expanded and not self.dropdowns:
                self._init_dropdowns()
            return True, False
        
        if not self.expanded:
            return False, False
        
        handlers = [
            ('iter', lambda v: setattr(self, 'max_iter', int(v))),
            ('func', self._on_func_change),
            ('escape', lambda v: setattr(self, 'escape_radius', float(v))),
            ('color', self._on_color_change),
        ]
        for name, on_change in handlers:
            dd = self.dropdowns.get(name)
            if dd:
                handled, changed = dd.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
                if handled:
                    if changed:
                        on_change(dd.value)
                    return True, changed
        
        if self.func_input:
            handled, _ = self.func_input.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
            if handled:
                return True, False
        
        if self.save_button_rect and self.save_button_rect.collidepoint(mx, my):
            self.save_requested = True
            return True, False
        
        if self.gpu_toggle_rect and self.gpu_toggle_rect.collidepoint(mx, my) and self.gpu_available:
            self.use_gpu = not self.use_gpu
            self.gpu_toggle_requested = True
            return True, False
        
        return self.get_rect().collidepoint(mx, my), False
    
    def _on_func_change(self, value):
        for name, fid in self.funcs:
            if name == value:
                self.func_id, self.func_display = fid, name
                self.custom_formula = self.formula_error = None
                if self.func_input:
                    self.func_input.text = ""
                break
    
    def _on_color_change(self, value):
        if value == 'Custom...':
            self.gradient_editor.show()
        else:
            self.colormap_name, self.using_custom = value, False
    
    def _handle_key(self, event):
        if not self.expanded or not self.func_input or not self.func_input.active:
            return False, False
        handled, changed = self.func_input.handle_event(event)
        if changed and self.formula_error:
            self.formula_error = None
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.func_input.text.strip():
            result, msg = parse_formula(self.func_input.text.strip())
            if result is not None:
                self.func_id, self.func_display, self.custom_formula = result, msg, None
            else:
                self.custom_formula, self.func_display = self.func_input.text.strip(), self.func_input.text.strip()
            self.formula_error = None
            return True, True
        return handled, False
    
    def draw(self, screen):
        self._init()
        
        # Toggle button
        pygame.draw.rect(screen, (60, 60, 60), (self.x, self.y, 120, 24))
        pygame.draw.rect(screen, (120, 120, 120), (self.x, self.y, 120, 24), 1)
        screen.blit(self.font.render('Toggle Settings', True, (200, 200, 200)), (self.x + 8, self.y + 5))
        
        if self.expanded:
            self._draw_expanded(screen)
        self.gradient_editor.draw(screen)
    
    def _draw_expanded(self, screen):
        rect = self.get_rect()
        pygame.draw.rect(screen, (40, 40, 40), rect)
        pygame.draw.rect(screen, (100, 100, 100), rect, 1)
        
        y = self.y + 10
        for label, key, extra in [
            ('Max Iterations:', 'iter', None),
            ('Function f(z):', 'func', 'input'),
            ('Escape Radius:', 'escape', None),
            ('Color Scheme:', 'color', 'preview'),
        ]:
            screen.blit(self.font.render(label, True, (180, 180, 180)), (self.x + 8, y))
            y += 18
            dd = self.dropdowns.get(key)
            if dd:
                dd.y = y
                dd.draw(screen, self.font)
                y += dd.height + (len(dd.options) * 22 if dd.expanded else 0)
            
            if extra == 'input' and self.func_input:
                y += 4
                screen.blit(self.font.render('or type formula:', True, (140, 140, 140)), (self.x + 8, y))
                y += 16
                self.func_input.x, self.func_input.y = self.x + 8, y
                self.func_input.draw(screen, self.font)
                y += self.func_input.height
                if self.formula_error:
                    screen.blit(self.font.render(self.formula_error, True, (255, 100, 100)), (self.x + 8, y + 2))
                    y += 14
            
            if extra == 'preview' and dd and not dd.expanded:
                y += 5
                cmap = self.get_colormap()
                pw = self.width - 16
                for px in range(pw):
                    pygame.draw.line(screen, tuple(cmap[int(px / pw * (len(cmap) - 1))]), (self.x + 8 + px, y), (self.x + 8 + px, y + 12))
                y += 18
            y += 10
        
        # GPU section
        screen.blit(self.font.render('GPU Acceleration:', True, (180, 180, 180)), (self.x + 8, y))
        y += 18
        self.gpu_toggle_rect = pygame.Rect(self.x + 8, y, self.width - 16, 26)
        
        if self.gpu_available:
            bg, border, tc = ((70, 120, 70), (100, 180, 100), (220, 255, 220)) if self.use_gpu else ((100, 70, 70), (150, 100, 100), (255, 220, 220))
            label = "GPU ON" if self.use_gpu else "GPU OFF (CPU)"
        else:
            bg, border, tc, label = (60, 60, 60), (100, 100, 100), (150, 150, 150), "GPU N/A"
        
        pygame.draw.rect(screen, bg, self.gpu_toggle_rect)
        pygame.draw.rect(screen, border, self.gpu_toggle_rect, 1)
        t = self.font.render(label, True, tc)
        screen.blit(t, (self.gpu_toggle_rect.centerx - t.get_width() // 2, self.gpu_toggle_rect.centery - t.get_height() // 2))
        y += 28
        screen.blit(self.font.render(self.gpu_device_name, True, (120, 120, 120)), (self.x + 8, y))
        y += 26
        
        # Save button
        self.save_button_rect = pygame.Rect(self.x + 8, y, self.width - 16, 26)
        pygame.draw.rect(screen, (70, 100, 70), self.save_button_rect)
        pygame.draw.rect(screen, (100, 150, 100), self.save_button_rect, 1)
        t = self.font.render('Save High-Res Image', True, (220, 255, 220))
        screen.blit(t, (self.save_button_rect.centerx - t.get_width() // 2, self.save_button_rect.centery - t.get_height() // 2))
    
    def point_in_menu(self, pos):
        if self.gradient_editor.visible:
            if pygame.Rect(self.gradient_editor.x, self.gradient_editor.y, self.gradient_editor.width, self.gradient_editor.height).collidepoint(pos):
                return True
        return self.get_rect().collidepoint(pos)
    
    def update_gpu_status(self, available, enabled, device_name):
        self.gpu_available, self.use_gpu = available, enabled
        self.gpu_device_name = device_name[:25] + "..." if len(device_name) > 28 else device_name
