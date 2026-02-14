"""
Interactive settings menu for the Mandelbrot visualizer.

Provides dropdown menus for max iterations and color scheme selection,
plus a custom gradient editor for creating your own color schemes.
"""

import re
import json
import os
import numpy as np
import pygame
from .colormaps import COLORMAPS, NUM_COLORS


# Load settings from JSON file
def load_settings():
    """Load settings from settings.json file."""
    settings_path = os.path.join(os.path.dirname(__file__), 'settings.json')
    try:
        with open(settings_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load settings.json: {e}")
        return None


# Global settings loaded from JSON
_SETTINGS = load_settings()


class TextInput:
    """A text input field component."""
    
    def __init__(self, x, y, width, height=24, placeholder="", initial_text=""):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = initial_text
        self.placeholder = placeholder
        self.active = False
        self.cursor_pos = len(initial_text)
        self.cursor_visible = True
        self.cursor_timer = 0
        
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def handle_event(self, event):
        """Returns (handled, text_changed)."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            rect = self.get_rect()
            was_active = self.active
            self.active = rect.collidepoint(mx, my)
            if self.active:
                # Position cursor near click
                return True, False
            elif was_active:
                return True, False
        
        elif event.type == pygame.KEYDOWN and self.active:
            old_text = self.text
            if event.key == pygame.K_BACKSPACE:
                if self.cursor_pos > 0:
                    self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                    self.cursor_pos -= 1
            elif event.key == pygame.K_DELETE:
                if self.cursor_pos < len(self.text):
                    self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]
            elif event.key == pygame.K_LEFT:
                self.cursor_pos = max(0, self.cursor_pos - 1)
            elif event.key == pygame.K_RIGHT:
                self.cursor_pos = min(len(self.text), self.cursor_pos + 1)
            elif event.key == pygame.K_HOME:
                self.cursor_pos = 0
            elif event.key == pygame.K_END:
                self.cursor_pos = len(self.text)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                self.active = False
                return True, old_text != self.text
            elif event.unicode and event.unicode.isprintable():
                self.text = self.text[:self.cursor_pos] + event.unicode + self.text[self.cursor_pos:]
                self.cursor_pos += 1
            
            return True, old_text != self.text
        
        return False, False
    
    def draw(self, screen, font, small_font):
        rect = self.get_rect()
        
        # Background
        bg_color = (60, 60, 70) if self.active else (50, 50, 55)
        pygame.draw.rect(screen, bg_color, rect)
        
        # Border
        border_color = (100, 140, 180) if self.active else (80, 80, 80)
        pygame.draw.rect(screen, border_color, rect, 1 if not self.active else 2)
        
        # Text or placeholder
        if self.text:
            text_surface = small_font.render(self.text, True, (220, 220, 220))
        else:
            text_surface = small_font.render(self.placeholder, True, (120, 120, 120))
        
        # Clip text to fit
        text_rect = text_surface.get_rect()
        text_rect.centery = rect.centery
        text_rect.left = rect.left + 6
        screen.blit(text_surface, text_rect)
        
        # Cursor
        if self.active:
            self.cursor_timer += 1
            if self.cursor_timer > 30:
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = 0
            
            if self.cursor_visible:
                # Calculate cursor x position
                if self.text:
                    pre_cursor = small_font.render(self.text[:self.cursor_pos], True, (220, 220, 220))
                    cursor_x = rect.left + 6 + pre_cursor.get_width()
                else:
                    cursor_x = rect.left + 6
                pygame.draw.line(screen, (220, 220, 220), 
                               (cursor_x, rect.top + 4), 
                               (cursor_x, rect.bottom - 4), 2)


class Dropdown:
    """A dropdown/select component."""
    
    def __init__(self, x, y, width, options, selected_idx=0, label=""):
        self.x = x
        self.y = y
        self.width = width
        self.height = 24
        self.options = options
        self.selected_idx = selected_idx
        self.label = label
        self.expanded = False
        self.hovered_idx = -1
        
    def get_value(self):
        return self.options[self.selected_idx]
    
    def set_value(self, value):
        if value in self.options:
            self.selected_idx = self.options.index(value)
    
    def get_rect(self):
        """Get the full rect including dropdown items when expanded."""
        if self.expanded:
            total_height = self.height + len(self.options) * 22
            return pygame.Rect(self.x, self.y, self.width, total_height)
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def handle_event(self, event):
        """Returns (handled, value_changed)."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            
            # Click on main dropdown button
            button_rect = pygame.Rect(self.x, self.y, self.width, self.height)
            if button_rect.collidepoint(mx, my):
                self.expanded = not self.expanded
                return True, False
            
            # Click on dropdown item
            if self.expanded:
                item_y = self.y + self.height
                for i, opt in enumerate(self.options):
                    item_rect = pygame.Rect(self.x, item_y, self.width, 22)
                    if item_rect.collidepoint(mx, my):
                        old_idx = self.selected_idx
                        self.selected_idx = i
                        self.expanded = False
                        return True, (old_idx != i)
                    item_y += 22
                
                # Click outside dropdown - close it
                self.expanded = False
                return True, False
        
        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            self.hovered_idx = -1
            if self.expanded:
                item_y = self.y + self.height
                for i in range(len(self.options)):
                    item_rect = pygame.Rect(self.x, item_y, self.width, 22)
                    if item_rect.collidepoint(mx, my):
                        self.hovered_idx = i
                    item_y += 22
        
        return False, False
    
    def draw(self, screen, font, small_font):
        # Main button
        button_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, (55, 55, 55), button_rect)
        pygame.draw.rect(screen, (100, 100, 100), button_rect, 1)
        
        # Selected value text
        text = small_font.render(str(self.get_value()), True, (220, 220, 220))
        screen.blit(text, (self.x + 8, self.y + 5))
        
        # Dropdown arrow
        arrow = "▼" if not self.expanded else "▲"
        arrow_text = small_font.render(arrow, True, (150, 150, 150))
        screen.blit(arrow_text, (self.x + self.width - 18, self.y + 5))
        
        # Dropdown items
        if self.expanded:
            item_y = self.y + self.height
            for i, opt in enumerate(self.options):
                item_rect = pygame.Rect(self.x, item_y, self.width, 22)
                
                if i == self.selected_idx:
                    pygame.draw.rect(screen, (70, 100, 70), item_rect)
                elif i == self.hovered_idx:
                    pygame.draw.rect(screen, (65, 65, 65), item_rect)
                else:
                    pygame.draw.rect(screen, (50, 50, 50), item_rect)
                
                pygame.draw.rect(screen, (80, 80, 80), item_rect, 1)
                
                color = (255, 255, 255) if i == self.selected_idx else (180, 180, 180)
                text = small_font.render(str(opt), True, color)
                screen.blit(text, (self.x + 8, item_y + 4))
                item_y += 22


def hsv_to_rgb(h, s, v):
    """Convert HSV (0-1 range) to RGB (0-255 range)."""
    if s == 0:
        r = g = b = int(v * 255)
        return (r, g, b)
    
    h = h * 6
    i = int(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    
    return (int(r * 255), int(g * 255), int(b * 255))


def rgb_to_hsv(r, g, b):
    """Convert RGB (0-255 range) to HSV (0-1 range)."""
    r, g, b = r / 255, g / 255, b / 255
    mx = max(r, g, b)
    mn = min(r, g, b)
    v = mx
    
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / (mx - mn)) + 360) % 360 / 360
    elif mx == g:
        h = (60 * ((b - r) / (mx - mn)) + 120) / 360
    else:
        h = (60 * ((r - g) / (mx - mn)) + 240) / 360
    
    s = 0 if mx == 0 else (mx - mn) / mx
    return (h, s, v)


class GradientEditor:
    """
    A popup gradient editor for creating custom color schemes.
    
    Features a Photoshop-style HSV color picker with saturation/value square
    and hue slider, plus smooth RGB sliders.
    """
    
    def __init__(self, screen_width, screen_height):
        self.width = 500
        self.height = 420
        self.x = (screen_width - self.width) // 2
        self.y = (screen_height - self.height) // 2
        
        self.visible = False
        self.font = None
        self.small_font = None
        
        # Gradient stops: list of (position, (r, g, b))
        self.stops = [
            (0.0, (0, 0, 0)),
            (0.5, (255, 100, 0)),
            (1.0, (255, 255, 255)),
        ]
        
        self.selected_stop = -1
        self.dragging_stop = -1
        self.editing_color = -1
        
        # Drag states for color picker
        self.dragging_sv = False      # Dragging on saturation/value square
        self.dragging_hue = False     # Dragging on hue bar
        self.dragging_slider = -1     # Which RGB slider is being dragged (-1 = none, 0=R, 1=G, 2=B)
        
        # Current HSV for the color picker (used when editing)
        self.current_h = 0.0
        self.current_s = 1.0
        self.current_v = 1.0
        
        # Gradient bar dimensions
        self.bar_x = self.x + 20
        self.bar_y = self.y + 60
        self.bar_width = self.width - 40
        self.bar_height = 30
        
        # Color picker dimensions
        self.picker_x = self.x + 20
        self.picker_y = self.y + 130
        self.sv_size = 150  # Saturation/Value square size
        self.hue_width = 20
        self.hue_height = 150
        
        # RGB slider dimensions
        self.slider_x = self.x + 210
        self.slider_y = self.y + 160
        self.slider_w = 180
        self.slider_h = 18
        
        # Pre-render surfaces for color picker
        self.sv_surface = None
        self.hue_surface = None
        
    def init_fonts(self):
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 14)
        self.small_font = pygame.font.SysFont('Arial', 12)
    
    def _create_hue_surface(self):
        """Create the hue bar surface (only needs to be done once)."""
        self.hue_surface = pygame.Surface((self.hue_width, self.hue_height))
        for y in range(self.hue_height):
            h = y / self.hue_height
            color = hsv_to_rgb(h, 1.0, 1.0)
            pygame.draw.line(self.hue_surface, color, (0, y), (self.hue_width, y))
    
    def _create_sv_surface(self):
        """Create the saturation/value square for current hue."""
        self.sv_surface = pygame.Surface((self.sv_size, self.sv_size))
        for x in range(self.sv_size):
            for y in range(self.sv_size):
                s = x / self.sv_size
                v = 1.0 - (y / self.sv_size)
                color = hsv_to_rgb(self.current_h, s, v)
                self.sv_surface.set_at((x, y), color)
    
    def show(self):
        self.visible = True
        self.selected_stop = -1
        self.dragging_stop = -1
        self.editing_color = -1
        self.dragging_sv = False
        self.dragging_hue = False
        self.dragging_slider = -1
        # Create hue surface if needed
        if self.hue_surface is None:
            self._create_hue_surface()
    
    def hide(self):
        self.visible = False
        self.dragging_sv = False
        self.dragging_hue = False
        self.dragging_slider = -1
    
    def _update_stop_color(self):
        """Update the selected stop's color from current HSV."""
        if self.editing_color >= 0 and self.editing_color < len(self.stops):
            rgb = hsv_to_rgb(self.current_h, self.current_s, self.current_v)
            self.stops[self.editing_color] = (self.stops[self.editing_color][0], rgb)
    
    def _sync_hsv_from_stop(self):
        """Sync current HSV from the selected stop's RGB."""
        if self.editing_color >= 0 and self.editing_color < len(self.stops):
            r, g, b = self.stops[self.editing_color][1]
            self.current_h, self.current_s, self.current_v = rgb_to_hsv(r, g, b)
            self._create_sv_surface()
    
    def get_colormap(self):
        """Generate a colormap array from the current gradient stops."""
        colors = np.zeros((NUM_COLORS, 3), dtype=np.uint8)
        sorted_stops = sorted(self.stops, key=lambda s: s[0])
        
        if len(sorted_stops) == 0:
            return colors
        if len(sorted_stops) == 1:
            colors[:] = sorted_stops[0][1]
            return colors
        
        for i in range(NUM_COLORS):
            t = i / (NUM_COLORS - 1)
            left_stop = sorted_stops[0]
            right_stop = sorted_stops[-1]
            
            for j in range(len(sorted_stops) - 1):
                if sorted_stops[j][0] <= t <= sorted_stops[j + 1][0]:
                    left_stop = sorted_stops[j]
                    right_stop = sorted_stops[j + 1]
                    break
            
            if right_stop[0] == left_stop[0]:
                local_t = 0
            else:
                local_t = (t - left_stop[0]) / (right_stop[0] - left_stop[0])
            
            for c in range(3):
                colors[i, c] = int(left_stop[1][c] * (1 - local_t) + right_stop[1][c] * local_t)
        
        return colors
    
    def handle_event(self, event):
        """Returns (handled, should_apply)."""
        if not self.visible:
            return False, False
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            
            # Check if click is outside the editor
            editor_rect = pygame.Rect(self.x, self.y, self.width, self.height)
            if not editor_rect.collidepoint(mx, my):
                self.hide()
                return True, False
            
            # Check Apply button
            apply_rect = pygame.Rect(self.x + self.width - 80, self.y + self.height - 40, 60, 28)
            if apply_rect.collidepoint(mx, my):
                self.hide()
                return True, True
            
            # Check Cancel button
            cancel_rect = pygame.Rect(self.x + self.width - 150, self.y + self.height - 40, 60, 28)
            if cancel_rect.collidepoint(mx, my):
                self.hide()
                return True, False
            
            # Check Add Stop button
            add_rect = pygame.Rect(self.x + 20, self.y + self.height - 40, 80, 28)
            if add_rect.collidepoint(mx, my):
                sorted_stops = sorted(self.stops, key=lambda s: s[0])
                new_pos = 0.5
                if len(sorted_stops) >= 2:
                    max_gap = 0
                    for i in range(len(sorted_stops) - 1):
                        gap = sorted_stops[i + 1][0] - sorted_stops[i][0]
                        if gap > max_gap:
                            max_gap = gap
                            new_pos = (sorted_stops[i][0] + sorted_stops[i + 1][0]) / 2
                cmap = self.get_colormap()
                idx = int(new_pos * (NUM_COLORS - 1))
                new_color = tuple(cmap[idx])
                self.stops.append((new_pos, new_color))
                return True, False
            
            # Check color picker interactions (only if editing a stop)
            if self.editing_color >= 0:
                # Saturation/Value square
                sv_rect = pygame.Rect(self.picker_x, self.picker_y, self.sv_size, self.sv_size)
                if sv_rect.collidepoint(mx, my):
                    self.dragging_sv = True
                    self.current_s = (mx - self.picker_x) / self.sv_size
                    self.current_v = 1.0 - (my - self.picker_y) / self.sv_size
                    self.current_s = max(0, min(1, self.current_s))
                    self.current_v = max(0, min(1, self.current_v))
                    self._update_stop_color()
                    return True, False
                
                # Hue bar
                hue_x = self.picker_x + self.sv_size + 10
                hue_rect = pygame.Rect(hue_x, self.picker_y, self.hue_width, self.hue_height)
                if hue_rect.collidepoint(mx, my):
                    self.dragging_hue = True
                    self.current_h = (my - self.picker_y) / self.hue_height
                    self.current_h = max(0, min(1, self.current_h))
                    self._create_sv_surface()
                    self._update_stop_color()
                    return True, False
                
                # RGB sliders
                for i in range(3):
                    sy = self.slider_y + i * 35
                    slider_rect = pygame.Rect(self.slider_x, sy, self.slider_w, self.slider_h)
                    if slider_rect.collidepoint(mx, my):
                        self.dragging_slider = i
                        val = int((mx - self.slider_x) / self.slider_w * 255)
                        val = max(0, min(255, val))
                        color = list(self.stops[self.editing_color][1])
                        color[i] = val
                        self.stops[self.editing_color] = (self.stops[self.editing_color][0], tuple(color))
                        self._sync_hsv_from_stop()
                        return True, False
            
            # Check gradient bar stops
            for i, (pos, color) in enumerate(self.stops):
                stop_x = self.bar_x + pos * self.bar_width
                stop_rect = pygame.Rect(stop_x - 8, self.bar_y + self.bar_height, 16, 18)
                if stop_rect.collidepoint(mx, my):
                    self.selected_stop = i
                    self.dragging_stop = i
                    self.editing_color = i
                    self._sync_hsv_from_stop()
                    return True, False
            
            # Click on gradient bar - deselect
            bar_rect = pygame.Rect(self.bar_x, self.bar_y, self.bar_width, self.bar_height)
            if bar_rect.collidepoint(mx, my):
                self.selected_stop = -1
                self.editing_color = -1
            
            return True, False
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging_stop = -1
            self.dragging_sv = False
            self.dragging_hue = False
            self.dragging_slider = -1
            return self.visible, False
        
        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            
            # Dragging gradient stop
            if self.dragging_stop >= 0:
                new_pos = (mx - self.bar_x) / self.bar_width
                new_pos = max(0.0, min(1.0, new_pos))
                self.stops[self.dragging_stop] = (new_pos, self.stops[self.dragging_stop][1])
                return True, False
            
            # Dragging on SV square
            if self.dragging_sv and self.editing_color >= 0:
                self.current_s = (mx - self.picker_x) / self.sv_size
                self.current_v = 1.0 - (my - self.picker_y) / self.sv_size
                self.current_s = max(0, min(1, self.current_s))
                self.current_v = max(0, min(1, self.current_v))
                self._update_stop_color()
                return True, False
            
            # Dragging on hue bar
            if self.dragging_hue and self.editing_color >= 0:
                self.current_h = (my - self.picker_y) / self.hue_height
                self.current_h = max(0, min(1, self.current_h))
                self._create_sv_surface()
                self._update_stop_color()
                return True, False
            
            # Dragging RGB slider
            if self.dragging_slider >= 0 and self.editing_color >= 0:
                val = int((mx - self.slider_x) / self.slider_w * 255)
                val = max(0, min(255, val))
                color = list(self.stops[self.editing_color][1])
                color[self.dragging_slider] = val
                self.stops[self.editing_color] = (self.stops[self.editing_color][0], tuple(color))
                self._sync_hsv_from_stop()
                return True, False
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                if self.selected_stop >= 0 and len(self.stops) > 2:
                    del self.stops[self.selected_stop]
                    self.selected_stop = -1
                    self.editing_color = -1
                    return True, False
        
        return self.visible, False
    
    def draw(self, screen):
        if not self.visible:
            return
        
        if self.font is None:
            self.init_fonts()
        
        # Background with border
        pygame.draw.rect(screen, (45, 45, 45), (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (100, 100, 100), (self.x, self.y, self.width, self.height), 2)
        
        # Title
        title = self.font.render('Custom Gradient Editor', True, (220, 220, 220))
        screen.blit(title, (self.x + 15, self.y + 10))
        
        # Instructions
        instr = self.small_font.render('Click stops to edit. Drag to move. Delete key to remove.', True, (150, 150, 150))
        screen.blit(instr, (self.x + 15, self.y + 32))
        
        # Gradient bar
        cmap = self.get_colormap()
        for px in range(self.bar_width):
            idx = int(px / self.bar_width * (NUM_COLORS - 1))
            color = tuple(cmap[idx])
            pygame.draw.line(screen, color,
                           (self.bar_x + px, self.bar_y),
                           (self.bar_x + px, self.bar_y + self.bar_height))
        pygame.draw.rect(screen, (100, 100, 100), 
                        (self.bar_x, self.bar_y, self.bar_width, self.bar_height), 1)
        
        # Draw stops
        for i, (pos, color) in enumerate(self.stops):
            stop_x = int(self.bar_x + pos * self.bar_width)
            points = [
                (stop_x, self.bar_y + self.bar_height),
                (stop_x - 8, self.bar_y + self.bar_height + 15),
                (stop_x + 8, self.bar_y + self.bar_height + 15),
            ]
            if i == self.selected_stop:
                pygame.draw.polygon(screen, (255, 255, 100), points)
                pygame.draw.polygon(screen, (255, 255, 255), points, 2)
            else:
                pygame.draw.polygon(screen, color, points)
                pygame.draw.polygon(screen, (200, 200, 200), points, 1)
        
        # Color picker for selected stop
        if self.editing_color >= 0 and self.editing_color < len(self.stops):
            pos, color = self.stops[self.editing_color]
            
            # Position label
            pos_label = self.small_font.render(f'Position: {pos:.2f}', True, (180, 180, 180))
            screen.blit(pos_label, (self.picker_x, self.picker_y - 20))
            
            # Saturation/Value square
            if self.sv_surface:
                screen.blit(self.sv_surface, (self.picker_x, self.picker_y))
            pygame.draw.rect(screen, (80, 80, 80), 
                           (self.picker_x, self.picker_y, self.sv_size, self.sv_size), 1)
            
            # SV cursor (crosshair)
            cursor_x = int(self.picker_x + self.current_s * self.sv_size)
            cursor_y = int(self.picker_y + (1.0 - self.current_v) * self.sv_size)
            pygame.draw.circle(screen, (255, 255, 255), (cursor_x, cursor_y), 6, 2)
            pygame.draw.circle(screen, (0, 0, 0), (cursor_x, cursor_y), 5, 1)
            
            # Hue bar
            hue_x = self.picker_x + self.sv_size + 10
            if self.hue_surface:
                screen.blit(self.hue_surface, (hue_x, self.picker_y))
            pygame.draw.rect(screen, (80, 80, 80), 
                           (hue_x, self.picker_y, self.hue_width, self.hue_height), 1)
            
            # Hue cursor (horizontal line)
            hue_cursor_y = int(self.picker_y + self.current_h * self.hue_height)
            pygame.draw.rect(screen, (255, 255, 255), 
                           (hue_x - 2, hue_cursor_y - 2, self.hue_width + 4, 4), 0)
            pygame.draw.rect(screen, (0, 0, 0), 
                           (hue_x - 2, hue_cursor_y - 2, self.hue_width + 4, 4), 1)
            
            # Color preview
            preview_x = self.slider_x
            preview_y = self.picker_y - 20
            preview_rect = pygame.Rect(preview_x, preview_y, 50, 30)
            pygame.draw.rect(screen, color, preview_rect)
            pygame.draw.rect(screen, (100, 100, 100), preview_rect, 1)
            
            # RGB values label
            rgb_label = self.small_font.render(f'R:{color[0]} G:{color[1]} B:{color[2]}', True, (180, 180, 180))
            screen.blit(rgb_label, (preview_x + 55, preview_y + 8))
            
            # RGB sliders with smooth thumb
            for i, (label, val) in enumerate(zip(['R', 'G', 'B'], color)):
                sy = self.slider_y + i * 35
                
                # Label
                lbl = self.small_font.render(f'{label}:', True, (180, 180, 180))
                screen.blit(lbl, (self.slider_x - 20, sy + 2))
                
                # Slider track (gradient showing color change)
                for sx in range(self.slider_w):
                    t = sx / self.slider_w
                    preview_color = list(color)
                    preview_color[i] = int(t * 255)
                    pygame.draw.line(screen, preview_color, 
                                   (self.slider_x + sx, sy),
                                   (self.slider_x + sx, sy + self.slider_h))
                
                pygame.draw.rect(screen, (80, 80, 80), 
                               (self.slider_x, sy, self.slider_w, self.slider_h), 1)
                
                # Slider thumb
                thumb_x = int(self.slider_x + val / 255 * self.slider_w)
                thumb_rect = pygame.Rect(thumb_x - 4, sy - 2, 8, self.slider_h + 4)
                pygame.draw.rect(screen, (220, 220, 220), thumb_rect)
                pygame.draw.rect(screen, (100, 100, 100), thumb_rect, 1)
                
                # Value text
                val_text = self.small_font.render(str(val), True, (220, 220, 220))
                screen.blit(val_text, (self.slider_x + self.slider_w + 10, sy + 2))
        
        # Buttons
        # Add Stop
        add_rect = pygame.Rect(self.x + 20, self.y + self.height - 40, 80, 28)
        pygame.draw.rect(screen, (60, 80, 60), add_rect)
        pygame.draw.rect(screen, (100, 120, 100), add_rect, 1)
        add_text = self.small_font.render('Add Stop', True, (220, 220, 220))
        screen.blit(add_text, (add_rect.x + 10, add_rect.y + 6))
        
        # Cancel
        cancel_rect = pygame.Rect(self.x + self.width - 150, self.y + self.height - 40, 60, 28)
        pygame.draw.rect(screen, (80, 60, 60), cancel_rect)
        pygame.draw.rect(screen, (120, 100, 100), cancel_rect, 1)
        cancel_text = self.small_font.render('Cancel', True, (220, 220, 220))
        screen.blit(cancel_text, (cancel_rect.x + 8, cancel_rect.y + 6))
        
        # Apply
        apply_rect = pygame.Rect(self.x + self.width - 80, self.y + self.height - 40, 60, 28)
        pygame.draw.rect(screen, (60, 80, 100), apply_rect)
        pygame.draw.rect(screen, (100, 120, 140), apply_rect, 1)
        apply_text = self.small_font.render('Apply', True, (220, 220, 220))
        screen.blit(apply_text, (apply_rect.x + 12, apply_rect.y + 6))


def parse_formula(formula):
    """
    Parse a user-entered formula and try to match it to a known function ID.
    
    Supports formulas like:
    - z^2 + c, z^3 + c, z^4 + c, etc.
    - z^2 + z + c, z^3 + z + c, etc.
    - z^2 - z + c, z^3 - z + c
    - c*sin(z), c*cos(z), c*e^z
    - conj(z)^2 + c or z_bar^2 + c for Tricorn
    - z^2 + c*z
    - z*(z^2 + c) or z^3 + c*z
    - z^2 + c^2
    - (z^2+c)/(z^2-c)
    
    Returns:
        (func_id, display_name) if matched, or (None, error_message) if not.
    """
    # Normalize the formula
    f = formula.lower().replace(' ', '').replace('·', '*').replace('×', '*')
    f = f.replace('^', '**').replace('²', '**2').replace('³', '**3')
    f = f.replace('⁴', '**4').replace('⁵', '**5').replace('⁶', '**6')
    f = f.replace('⁷', '**7').replace('⁸', '**8')
    f = f.replace('z_bar', 'conj(z)').replace('z̄', 'conj(z)')
    
    # Dictionary of patterns to match (using ASCII display names)
    patterns = [
        # z^n + c patterns
        (r'^z\*\*2\+c$', 0, 'z^2 + c'),
        (r'^z\*\*3\+c$', 1, 'z^3 + c'),
        (r'^z\*\*4\+c$', 2, 'z^4 + c'),
        (r'^z\*\*5\+c$', 3, 'z^5 + c'),
        (r'^z\*\*6\+c$', 8, 'z^6 + c'),
        (r'^z\*\*7\+c$', 9, 'z^7 + c'),
        (r'^z\*\*8\+c$', 10, 'z^8 + c'),
        
        # Tricorn
        (r'^conj\(z\)\*\*2\+c$', 4, 'conj(z)^2 + c'),
        
        # Transcendental - support multiple notations
        (r'^c\*e\*\*z$', 5, 'c*exp(z)'),
        (r'^c\*exp\(z\)$', 5, 'c*exp(z)'),
        (r'^e\*\*z\+c$', 5, 'c*exp(z)'),
        (r'^exp\(z\)\+c$', 5, 'c*exp(z)'),
        (r'^c\*sin\(z\)$', 6, 'c*sin(z)'),
        (r'^sin\(z\)\+c$', 6, 'c*sin(z)'),
        (r'^c\*cos\(z\)$', 14, 'c*cos(z)'),
        (r'^cos\(z\)\+c$', 14, 'c*cos(z)'),
        
        # z^n + z + c patterns
        (r'^z\*\*2\+z\+c$', 7, 'z^2 + z + c'),
        (r'^z\*\*3\+z\+c$', 12, 'z^3 + z + c'),
        (r'^z\*\*4\+z\+c$', 13, 'z^4 + z + c'),
        
        # z^n - z + c patterns
        (r'^z\*\*2-z\+c$', 15, 'z^2 - z + c'),
        (r'^z\*\*3-z\+c$', 19, 'z^3 - z + c'),
        
        # Other variants
        (r'^z\*\*2\+c\*z$', 11, 'z^2 + c*z'),
        (r'^z\*\*3\+c\*z$', 16, 'z^3 + c*z'),
        (r'^z\*\(z\*\*2\+c\)$', 16, 'z*(z^2 + c)'),
        (r'^z\*\*2\+c\*\*2$', 17, 'z^2 + c^2'),
        (r'^\(z\*\*2\+c\)/\(z\*\*2-c\)$', 18, '(z^2+c)/(z^2-c)'),
    ]
    
    for pattern, func_id, display in patterns:
        if re.match(pattern, f):
            return func_id, display
    
    # Try to extract z^n + c pattern for any n
    match = re.match(r'^z\*\*(\d+)\+c$', f)
    if match:
        n = int(match.group(1))
        if n == 2:
            return 0, 'z^2 + c'
        elif n == 3:
            return 1, 'z^3 + c'
        elif n == 4:
            return 2, 'z^4 + c'
        elif n == 5:
            return 3, 'z^5 + c'
        elif n == 6:
            return 8, 'z^6 + c'
        elif n == 7:
            return 9, 'z^7 + c'
        elif n == 8:
            return 10, 'z^8 + c'
        else:
            return None, f'z^{n} not supported (max 8)'
    
    return None, 'Unrecognized formula'


class Menu:
    """
    Settings menu with dropdowns for max iterations, color scheme,
    iteration function, escape radius, plus a custom gradient editor.
    Also supports typing custom formulas.
    """
    
    # Load options from settings.json or use defaults
    ITER_OPTIONS = _SETTINGS['iteration_options'] if _SETTINGS else [64, 128, 256, 512, 1024, 2048]
    ESCAPE_OPTIONS = _SETTINGS['escape_radius_options'] if _SETTINGS else [2, 4, 8, 16, 32, 64, 128, 256]
    
    # Iteration functions: (display_name, func_id) - loaded from settings.json
    if _SETTINGS:
        ITERATION_FUNCTIONS = [(f['name'], f['id']) for f in _SETTINGS['iteration_functions']]
    else:
        ITERATION_FUNCTIONS = [
            ('z^2 + c', 0),
            ('z^3 + c', 1),
            ('z^4 + c', 2),
            ('z^5 + c', 3),
            ('z^6 + c', 8),
            ('z^7 + c', 9),
            ('z^8 + c', 10),
            ('conj(z)^2 + c', 4),
            ('c*exp(z)', 5),
            ('c*sin(z)', 6),
            ('c*cos(z)', 14),
            ('z^2 + z + c', 7),
            ('z^3 + z + c', 12),
            ('z^4 + z + c', 13),
            ('z^2 - z + c', 15),
            ('z^3 - z + c', 19),
            ('z^2 + c*z', 11),
            ('z^3 + c*z', 16),
            ('z^2 + c^2', 17),
            ('(z^2+c)/(z^2-c)', 18),
        ]
    
    def __init__(self, x, y, width=220, screen_width=800, screen_height=800):
        self.x = x
        self.y = y
        self.width = width
        self.expanded = False
        
        # Fonts
        self.font = None
        self.small_font = None
        
        # Current settings
        self.max_iter = 512
        self.colormap_name = 'Hot'
        self.colormap_names = list(COLORMAPS.keys()) + ['Custom...']
        self.func_id = 0
        self.func_display = 'z^2 + c'
        self.escape_radius = 2.0
        
        # Custom formula support
        self.custom_formula = None  # When set, overrides func_id
        
        # Custom colormap (generated from gradient editor)
        self.custom_colormap = None
        self.using_custom = False
        
        # Dropdowns (initialized after menu expands)
        self.iter_dropdown = None
        self.color_dropdown = None
        self.func_dropdown = None
        self.escape_dropdown = None
        
        # Text input for custom function formula
        self.func_input = None
        self.formula_error = None  # Error message for invalid formula
        
        # Save image request flag
        self.save_requested = False
        
        # Gradient editor
        self.gradient_editor = GradientEditor(screen_width, screen_height)
        
        # UI state
        self.hovered_item = None
    
    def init_fonts(self):
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 14)
        self.small_font = pygame.font.SysFont('Arial', 12)
    
    def _init_dropdowns(self):
        """Initialize dropdowns and text inputs with current positions."""
        dropdown_width = self.width - 16
        
        # Iterations dropdown
        iter_options = [str(v) for v in self.ITER_OPTIONS]
        selected_iter = iter_options.index(str(self.max_iter)) if str(self.max_iter) in iter_options else 0
        self.iter_dropdown = Dropdown(
            self.x + 8, self.y + 28,
            dropdown_width, iter_options, selected_iter
        )
        
        # Function dropdown - find selected index by func_id
        func_names = [name for name, _ in self.ITERATION_FUNCTIONS]
        selected_func = 0
        for idx, (name, fid) in enumerate(self.ITERATION_FUNCTIONS):
            if fid == self.func_id:
                selected_func = idx
                break
        self.func_dropdown = Dropdown(
            self.x + 8, self.y + 75,
            dropdown_width, func_names, selected_func
        )
        
        # Function text input for custom formulas
        self.func_input = TextInput(
            self.x + 8, self.y + 105,
            dropdown_width, 24,
            placeholder="Type formula: z^3+c",
            initial_text=""
        )
        
        # Escape radius dropdown
        escape_options = [str(v) for v in self.ESCAPE_OPTIONS]
        selected_escape = escape_options.index(str(int(self.escape_radius))) if str(int(self.escape_radius)) in escape_options else 0
        self.escape_dropdown = Dropdown(
            self.x + 8, self.y + 152,
            dropdown_width, escape_options, selected_escape
        )
        
        # Color scheme dropdown
        selected_color = self.colormap_names.index(self.colormap_name) if self.colormap_name in self.colormap_names else 0
        self.color_dropdown = Dropdown(
            self.x + 8, self.y + 199,
            dropdown_width, self.colormap_names, selected_color
        )
    
    def get_rect(self):
        """Get the bounding rectangle of the menu."""
        if not self.expanded:
            return pygame.Rect(self.x, self.y, 120, 24)
        
        # Base height for all sections: iter, func dropdown + text input, escape, color + preview, save button
        # iter: ~40, func dropdown: ~50, text input: ~45, escape: ~40, color: ~55, save button: ~35
        base_height = 325
        
        # Add extra if there's a formula error
        if self.formula_error:
            base_height += 15
        
        # Add height for expanded dropdowns
        if self.iter_dropdown and self.iter_dropdown.expanded:
            base_height += len(self.iter_dropdown.options) * 22
        if self.func_dropdown and self.func_dropdown.expanded:
            base_height += len(self.func_dropdown.options) * 22
        if self.escape_dropdown and self.escape_dropdown.expanded:
            base_height += len(self.escape_dropdown.options) * 22
        if self.color_dropdown and self.color_dropdown.expanded:
            base_height += len(self.color_dropdown.options) * 22
        
        return pygame.Rect(self.x, self.y, self.width, base_height)
    
    def get_colormap(self):
        """Get the current colormap array."""
        if self.using_custom and self.custom_colormap is not None:
            return self.custom_colormap
        return COLORMAPS[self.colormap_name]()
    
    def _update_positions(self):
        """Update dropdown and input positions based on current layout."""
        if not self.expanded:
            return
        
        current_y = self.y + 10
        
        # Max Iterations section
        current_y += 18  # label
        if self.iter_dropdown:
            self.iter_dropdown.y = current_y
            current_y += self.iter_dropdown.height
            if self.iter_dropdown.expanded:
                current_y += len(self.iter_dropdown.options) * 22
        
        current_y += 10  # spacing
        
        # Function section
        current_y += 18  # label
        if self.func_dropdown:
            self.func_dropdown.y = current_y
            current_y += self.func_dropdown.height
            if self.func_dropdown.expanded:
                current_y += len(self.func_dropdown.options) * 22
        
        # Text input section
        current_y += 4
        current_y += 16  # "or type formula:" label
        if self.func_input:
            self.func_input.y = current_y
            current_y += self.func_input.height
        
        if self.formula_error:
            current_y += 14
        
        current_y += 8  # spacing
        
        # Escape radius section
        current_y += 18  # label
        if self.escape_dropdown:
            self.escape_dropdown.y = current_y
            current_y += self.escape_dropdown.height
            if self.escape_dropdown.expanded:
                current_y += len(self.escape_dropdown.options) * 22
        
        current_y += 10  # spacing
        
        # Color scheme section
        current_y += 18  # label
        if self.color_dropdown:
            self.color_dropdown.y = current_y
    
    def handle_event(self, event):
        """
        Handle a pygame event.
        Returns (handled, need_recompute).
        """
        # Update positions before handling events
        self._update_positions()
        
        # Gradient editor takes priority
        if self.gradient_editor.visible:
            handled, should_apply = self.gradient_editor.handle_event(event)
            if should_apply:
                self.custom_colormap = self.gradient_editor.get_colormap()
                self.using_custom = True
                self.colormap_name = 'Custom...'
                if self.color_dropdown:
                    self.color_dropdown.set_value('Custom...')
                return True, True
            return handled, False
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            
            # Toggle button
            toggle_rect = pygame.Rect(self.x, self.y, 120, 24)
            if toggle_rect.collidepoint(mx, my):
                self.expanded = not self.expanded
                if self.expanded and self.iter_dropdown is None:
                    self._init_dropdowns()
                return True, False
            
            if self.expanded:
                # Handle dropdowns in order (top to bottom)
                # Iterations
                if self.iter_dropdown:
                    handled, changed = self.iter_dropdown.handle_event(event)
                    if handled:
                        if changed:
                            self.max_iter = int(self.iter_dropdown.get_value())
                        return True, changed
                
                # Function dropdown
                if self.func_dropdown:
                    handled, changed = self.func_dropdown.handle_event(event)
                    if handled:
                        if changed:
                            selected_idx = self.func_dropdown.selected_idx
                            self.func_id = self.ITERATION_FUNCTIONS[selected_idx][1]
                            self.func_display = self.ITERATION_FUNCTIONS[selected_idx][0]
                            self.custom_formula = None  # Clear custom formula when using dropdown
                            self.formula_error = None
                            if self.func_input:
                                self.func_input.text = ""
                        return True, changed
                
                # Function text input
                if self.func_input:
                    handled, changed = self.func_input.handle_event(event)
                    if handled:
                        return True, False
                
                # Escape radius
                if self.escape_dropdown:
                    handled, changed = self.escape_dropdown.handle_event(event)
                    if handled:
                        if changed:
                            self.escape_radius = float(self.escape_dropdown.get_value())
                        return True, changed
                
                # Color scheme
                if self.color_dropdown:
                    handled, changed = self.color_dropdown.handle_event(event)
                    if handled:
                        if changed:
                            new_name = self.color_dropdown.get_value()
                            if new_name == 'Custom...':
                                self.gradient_editor.show()
                                return True, False
                            else:
                                self.colormap_name = new_name
                                self.using_custom = False
                        return True, changed
                
                # Save button
                if hasattr(self, 'save_button_rect') and self.save_button_rect.collidepoint(mx, my):
                    self.save_requested = True
                    return True, False
                
                # Click in menu area but not on dropdown
                menu_rect = self.get_rect()
                if menu_rect.collidepoint(mx, my):
                    return True, False
        
        elif event.type == pygame.KEYDOWN:
            # Handle function text input
            if self.expanded and self.func_input and self.func_input.active:
                handled, changed = self.func_input.handle_event(event)
                if handled:
                    # Clear error when user starts typing again
                    if changed and self.formula_error:
                        self.formula_error = None
                    # Check if Enter was pressed
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        if self.func_input.text.strip():
                            formula_text = self.func_input.text.strip()
                            result, msg = parse_formula(formula_text)
                            if result is not None:
                                # Matched a known function - use optimized JIT version
                                self.func_id = result
                                self.func_display = msg
                                self.custom_formula = None  # Clear custom
                                self.formula_error = None
                                return True, True
                            else:
                                # Use as custom formula (slower but flexible)
                                self.custom_formula = formula_text
                                self.func_display = formula_text
                                self.formula_error = None  # No error - custom formulas are allowed
                                return True, True
                    return True, False
        
        elif event.type == pygame.MOUSEMOTION:
            if self.expanded:
                if self.iter_dropdown:
                    self.iter_dropdown.handle_event(event)
                if self.func_dropdown:
                    self.func_dropdown.handle_event(event)
                if self.escape_dropdown:
                    self.escape_dropdown.handle_event(event)
                if self.color_dropdown:
                    self.color_dropdown.handle_event(event)
        
        return False, False
    
    def draw(self, screen):
        if self.font is None:
            self.init_fonts()
        
        self._draw_toggle_button(screen)
        
        if self.expanded:
            self._draw_expanded_menu(screen)
        
        # Draw gradient editor on top
        self.gradient_editor.draw(screen)
    
    def _draw_toggle_button(self, screen):
        toggle_rect = pygame.Rect(self.x, self.y, 120, 24)
        pygame.draw.rect(screen, (60, 60, 60), toggle_rect)
        pygame.draw.rect(screen, (120, 120, 120), toggle_rect, 1)
        
        toggle_text = self.font.render('Toggle Settings', True, (200, 200, 200))
        screen.blit(toggle_text, (self.x + 8, self.y + 4))
    
    def _draw_expanded_menu(self, screen):
        menu_rect = self.get_rect()
        pygame.draw.rect(screen, (40, 40, 40), menu_rect)
        pygame.draw.rect(screen, (100, 100, 100), menu_rect, 1)
        
        # Track vertical offset for stacking dropdowns
        current_y = self.y + 10
        
        # ---- Max Iterations ----
        label = self.small_font.render('Max Iterations:', True, (180, 180, 180))
        screen.blit(label, (self.x + 8, current_y))
        current_y += 18
        
        if self.iter_dropdown:
            self.iter_dropdown.y = current_y
            self.iter_dropdown.draw(screen, self.font, self.small_font)
            current_y += self.iter_dropdown.height
            if self.iter_dropdown.expanded:
                current_y += len(self.iter_dropdown.options) * 22
        
        current_y += 10  # spacing
        
        # ---- Iteration Function ----
        label = self.small_font.render('Function f(z):', True, (180, 180, 180))
        screen.blit(label, (self.x + 8, current_y))
        current_y += 18
        
        if self.func_dropdown:
            self.func_dropdown.y = current_y
            self.func_dropdown.draw(screen, self.font, self.small_font)
            current_y += self.func_dropdown.height
            if self.func_dropdown.expanded:
                current_y += len(self.func_dropdown.options) * 22
        
        # Custom function text input
        current_y += 4
        label = self.small_font.render('or type formula:', True, (140, 140, 140))
        screen.blit(label, (self.x + 8, current_y))
        current_y += 16
        
        if self.func_input:
            self.func_input.x = self.x + 8
            self.func_input.y = current_y
            self.func_input.draw(screen, self.font, self.small_font)
            current_y += self.func_input.height
        
        # Show formula error if any
        if self.formula_error:
            error_label = self.small_font.render(self.formula_error, True, (255, 100, 100))
            screen.blit(error_label, (self.x + 8, current_y + 2))
            current_y += 14
        
        current_y += 8  # spacing
        
        # ---- Escape Radius ----
        label = self.small_font.render('Escape Radius:', True, (180, 180, 180))
        screen.blit(label, (self.x + 8, current_y))
        current_y += 18
        
        if self.escape_dropdown:
            self.escape_dropdown.y = current_y
            self.escape_dropdown.draw(screen, self.font, self.small_font)
            current_y += self.escape_dropdown.height
            if self.escape_dropdown.expanded:
                current_y += len(self.escape_dropdown.options) * 22
        
        current_y += 10  # spacing
        
        # ---- Color Scheme ----
        label = self.small_font.render('Color Scheme:', True, (180, 180, 180))
        screen.blit(label, (self.x + 8, current_y))
        current_y += 18
        
        if self.color_dropdown:
            self.color_dropdown.y = current_y
            self.color_dropdown.draw(screen, self.font, self.small_font)
            current_y += self.color_dropdown.height
            if self.color_dropdown.expanded:
                current_y += len(self.color_dropdown.options) * 22
            
            # Draw color preview for selected scheme (only if dropdown not expanded)
            if not self.color_dropdown.expanded:
                current_y += 5
                cmap = self.get_colormap()
                preview_x = self.x + 8
                preview_w = self.width - 16
                for px in range(preview_w):
                    idx = int(px / preview_w * (len(cmap) - 1))
                    color = tuple(cmap[idx])
                    pygame.draw.line(screen, color,
                                   (preview_x + px, current_y),
                                   (preview_x + px, current_y + 12))
                current_y += 18
        
        # ---- Save High-Res Image Button ----
        current_y += 10  # spacing
        self.save_button_rect = pygame.Rect(self.x + 8, current_y, self.width - 16, 26)
        pygame.draw.rect(screen, (70, 100, 70), self.save_button_rect)
        pygame.draw.rect(screen, (100, 150, 100), self.save_button_rect, 1)
        save_text = self.font.render('Save High-Res Image', True, (220, 255, 220))
        text_x = self.save_button_rect.x + (self.save_button_rect.width - save_text.get_width()) // 2
        text_y = self.save_button_rect.y + (self.save_button_rect.height - save_text.get_height()) // 2
        screen.blit(save_text, (text_x, text_y))
    
    def point_in_menu(self, pos):
        """Check if a point is inside the menu or gradient editor."""
        if self.gradient_editor.visible:
            editor_rect = pygame.Rect(
                self.gradient_editor.x, self.gradient_editor.y,
                self.gradient_editor.width, self.gradient_editor.height
            )
            if editor_rect.collidepoint(pos):
                return True
        return self.get_rect().collidepoint(pos)
