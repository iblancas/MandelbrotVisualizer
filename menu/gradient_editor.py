"""
Gradient editor component for custom colormap creation.

This module provides a popup editor that allows users to create custom
color gradients for the fractal visualization. It features:

- Visual gradient preview with color stops
- Drag-and-drop stop positioning
- HSV color picker with saturation/value square
- RGB sliders for precise color adjustment
- Add/delete color stops

The editor creates smooth gradients by interpolating between color stops,
which are then converted to the 256-color colormap used by the renderer.
"""

import colorsys
import numpy as np
import pygame

from colormaps import NUM_COLORS


def _hsv_to_rgb(h, s, v):
    """
    Convert HSV color to RGB tuple.
    
    Args:
        h: Hue (0-1)
        s: Saturation (0-1)
        v: Value/brightness (0-1)
        
    Returns:
        Tuple of (r, g, b) with values 0-255
    """
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _rgb_to_hsv(r, g, b):
    """
    Convert RGB color to HSV tuple.
    
    Args:
        r: Red (0-255)
        g: Green (0-255)
        b: Blue (0-255)
        
    Returns:
        Tuple of (h, s, v) with values 0-1
    """
    return colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)


class GradientEditor:
    """
    Popup gradient editor for creating custom color schemes.
    
    The editor displays a gradient bar with draggable color stops.
    Clicking a stop opens a color picker with HSV and RGB controls.
    The resulting gradient is sampled to create a 256-color colormap.
    
    Gradient Algorithm:
    1. Sort stops by position (0.0 to 1.0)
    2. For each output color index (0 to 255):
       - Calculate position t in [0, 1]
       - Find left and right stops that bracket t
       - Linearly interpolate RGB between the two stops
    
    Attributes:
        visible: Whether the editor popup is shown
        stops: List of (position, (r, g, b)) tuples
        selected_stop: Index of currently selected stop (-1 for none)
        editing_color: Index of stop being color-edited (-1 for none)
    """
    
    def __init__(self, screen_width, screen_height):
        """
        Initialize the gradient editor.
        
        The editor is centered on screen.
        
        Args:
            screen_width: Screen width for centering
            screen_height: Screen height for centering
        """
        self.width = 500
        self.height = 420
        self.x = (screen_width - self.width) // 2
        self.y = (screen_height - self.height) // 2
        self.visible = False
        self.font = None
        
        # Gradient stops: list of (position, (r, g, b)) tuples
        # Position is 0.0 to 1.0, representing position along gradient
        self.stops = [
            (0.0, (0, 0, 0)),        # Black at start
            (0.5, (255, 100, 0)),    # Orange in middle
            (1.0, (255, 255, 255))   # White at end
        ]
        
        # Selection state
        self.selected_stop = -1    # Currently selected stop
        self.dragging_stop = -1    # Stop being dragged
        self.editing_color = -1    # Stop whose color is being edited
        
        # Color picker state
        self.dragging_sv = False   # Dragging in S/V square
        self.dragging_hue = False  # Dragging in hue bar
        self.dragging_slider = -1  # RGB slider being dragged (0, 1, or 2)
        self.hsv = [0.0, 1.0, 1.0] # Current color in HSV
        
        # Cached surfaces for color picker
        self.sv_surface = None     # Saturation/Value square
        self.hue_surface = None    # Hue rainbow bar
    
    def _init(self):
        """Initialize font on first use."""
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 12)
    
    def _bar_rect(self):
        """Get the gradient bar rectangle: (x, y, width, height)."""
        return self.x + 20, self.y + 60, self.width - 40, 30
    
    def _picker_rect(self):
        """Get the color picker SV square rectangle: (x, y, size, size)."""
        return self.x + 20, self.y + 130, 150, 150
    
    def _create_hue_surface(self):
        """
        Create the hue rainbow bar surface.
        
        This is a vertical gradient from red (top) through all hues
        to red again (bottom would be 1.0, shown truncated at 0.99).
        """
        self.hue_surface = pygame.Surface((20, 150))
        for y in range(150):
            hue = y / 150  # 0 at top to 1 at bottom
            color = _hsv_to_rgb(hue, 1, 1)  # Full saturation and value
            pygame.draw.line(self.hue_surface, color, (0, y), (20, y))
    
    def _create_sv_surface(self):
        """
        Create the saturation/value picker surface.
        
        This is a 2D gradient:
        - X axis: Saturation (0 at left to 1 at right)
        - Y axis: Value (1 at top to 0 at bottom)
        - Uses current hue from self.hsv[0]
        """
        self.sv_surface = pygame.Surface((150, 150))
        h = self.hsv[0]
        for x in range(150):
            for y in range(150):
                s = x / 150       # Saturation: left=0, right=1
                v = 1 - y / 150   # Value: top=1, bottom=0
                self.sv_surface.set_at((x, y), _hsv_to_rgb(h, s, v))
    
    def show(self):
        """Show the gradient editor popup."""
        self.visible = True
        self.selected_stop = -1
        self.dragging_stop = -1
        self.editing_color = -1
        self.dragging_sv = False
        self.dragging_hue = False
        self.dragging_slider = -1
        
        if not self.hue_surface:
            self._create_hue_surface()
    
    def hide(self):
        """Hide the gradient editor popup."""
        self.visible = False
    
    def _sync_color(self):
        """Update stop color from current HSV values."""
        if 0 <= self.editing_color < len(self.stops):
            pos = self.stops[self.editing_color][0]
            self.stops[self.editing_color] = (pos, _hsv_to_rgb(*self.hsv))
    
    def _sync_hsv(self):
        """
        Update HSV from selected stop's RGB color.
        
        Called when selecting a stop to edit - synchronizes the
        color picker to show the stop's current color.
        """
        if 0 <= self.editing_color < len(self.stops):
            rgb = self.stops[self.editing_color][1]
            self.hsv = list(_rgb_to_hsv(*rgb))
            self._create_sv_surface()
    
    def get_colormap(self):
        """
        Generate a 256-color colormap from the gradient stops.
        
        Uses linear interpolation between stops to create smooth
        color transitions.
        
        Returns:
            NumPy array of shape (256, 3) with RGB values
        """
        colors = np.zeros((NUM_COLORS, 3), dtype=np.uint8)
        stops = sorted(self.stops, key=lambda s: s[0])
        
        if len(stops) == 0:
            return colors  # All black
        
        if len(stops) == 1:
            colors[:] = stops[0][1]  # Solid color
            return colors
        
        # Interpolate between stops
        for i in range(NUM_COLORS):
            t = i / (NUM_COLORS - 1)  # Position in colormap (0 to 1)
            
            # Find stops that bracket this position
            left_stop = stops[0]
            right_stop = stops[-1]
            
            for j in range(len(stops) - 1):
                if stops[j][0] <= t <= stops[j + 1][0]:
                    left_stop = stops[j]
                    right_stop = stops[j + 1]
                    break
            
            # Interpolation factor between the two stops
            if right_stop[0] == left_stop[0]:
                lerp_t = 0
            else:
                lerp_t = (t - left_stop[0]) / (right_stop[0] - left_stop[0])
            
            # Linear interpolation of RGB values
            for c in range(3):
                colors[i, c] = int(
                    left_stop[1][c] * (1 - lerp_t) + right_stop[1][c] * lerp_t
                )
        
        return colors
    
    def handle_event(self, event):
        """
        Handle pygame events for the gradient editor.
        
        Args:
            event: Pygame event to process
            
        Returns:
            Tuple of (handled: bool, should_apply: bool)
            should_apply is True when user clicks Apply button
        """
        if not self.visible:
            return False, False
        
        bx, by, bw, bh = self._bar_rect()
        px, py, ps, _ = self._picker_rect()
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            
            # Click outside editor closes it
            if not pygame.Rect(self.x, self.y, self.width, self.height).collidepoint(mx, my):
                self.hide()
                return True, False
            
            # Apply button
            if pygame.Rect(self.x + self.width - 80, self.y + self.height - 40, 60, 28).collidepoint(mx, my):
                self.hide()
                return True, True  # Apply the gradient
            
            # Cancel button
            if pygame.Rect(self.x + self.width - 150, self.y + self.height - 40, 60, 28).collidepoint(mx, my):
                self.hide()
                return True, False  # Discard changes
            
            # Add Stop button
            if pygame.Rect(self.x + 20, self.y + self.height - 40, 80, 28).collidepoint(mx, my):
                self._add_stop()
                return True, False
            
            # Color picker interactions (only when editing a stop)
            if self.editing_color >= 0:
                # Saturation/Value square
                if pygame.Rect(px, py, ps, ps).collidepoint(mx, my):
                    self.dragging_sv = True
                    self.hsv[1] = max(0, min(1, (mx - px) / ps))
                    self.hsv[2] = max(0, min(1, 1 - (my - py) / ps))
                    self._sync_color()
                    return True, False
                
                # Hue bar
                hue_x = px + ps + 10
                if pygame.Rect(hue_x, py, 20, 150).collidepoint(mx, my):
                    self.dragging_hue = True
                    self.hsv[0] = max(0, min(1, (my - py) / 150))
                    self._create_sv_surface()
                    self._sync_color()
                    return True, False
                
                # RGB sliders
                slider_x = self.x + 210
                slider_y = self.y + 160
                for i in range(3):
                    if pygame.Rect(slider_x, slider_y + i * 35, 180, 18).collidepoint(mx, my):
                        self.dragging_slider = i
                        self._update_slider(mx)
                        return True, False
            
            # Click on gradient stops
            for i, (pos, _) in enumerate(self.stops):
                stop_x = bx + pos * bw
                stop_rect = pygame.Rect(stop_x - 8, by + bh, 16, 18)
                if stop_rect.collidepoint(mx, my):
                    self.selected_stop = i
                    self.dragging_stop = i
                    self.editing_color = i
                    self._sync_hsv()
                    return True, False
            
            # Click on gradient bar (not on a stop) - deselect
            if pygame.Rect(bx, by, bw, bh).collidepoint(mx, my):
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
            
            # Drag stop position
            if self.dragging_stop >= 0:
                pos = max(0, min(1, (mx - bx) / bw))
                self.stops[self.dragging_stop] = (pos, self.stops[self.dragging_stop][1])
                return True, False
            
            # Drag in S/V square
            if self.dragging_sv and self.editing_color >= 0:
                self.hsv[1] = max(0, min(1, (mx - px) / ps))
                self.hsv[2] = max(0, min(1, 1 - (my - py) / ps))
                self._sync_color()
                return True, False
            
            # Drag on hue bar
            if self.dragging_hue and self.editing_color >= 0:
                self.hsv[0] = max(0, min(1, (my - py) / 150))
                self._create_sv_surface()
                self._sync_color()
                return True, False
            
            # Drag RGB slider
            if self.dragging_slider >= 0 and self.editing_color >= 0:
                self._update_slider(mx)
                return True, False
        
        elif event.type == pygame.KEYDOWN:
            # Delete key removes selected stop (if more than 2 remain)
            if event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                if self.selected_stop >= 0 and len(self.stops) > 2:
                    del self.stops[self.selected_stop]
                    self.selected_stop = -1
                    self.editing_color = -1
                return True, False
        
        return self.visible, False
    
    def _add_stop(self):
        """
        Add a new color stop at the largest gap.
        
        Finds the largest gap between existing stops and places
        the new stop in the middle, with interpolated color.
        """
        sorted_stops = sorted(self.stops, key=lambda x: x[0])
        
        # Find largest gap
        pos = 0.5  # Default to middle
        if len(sorted_stops) >= 2:
            max_gap = 0
            for i in range(len(sorted_stops) - 1):
                gap = sorted_stops[i + 1][0] - sorted_stops[i][0]
                if gap > max_gap:
                    max_gap = gap
                    pos = (sorted_stops[i][0] + sorted_stops[i + 1][0]) / 2
        
        # Get interpolated color at this position
        colormap = self.get_colormap()
        color_idx = int(pos * (NUM_COLORS - 1))
        color = tuple(colormap[color_idx])
        
        self.stops.append((pos, color))
    
    def _update_slider(self, mx):
        """
        Update color from RGB slider drag.
        
        Args:
            mx: Mouse X position
        """
        slider_x = self.x + 210
        val = max(0, min(255, int((mx - slider_x) / 180 * 255)))
        
        color = list(self.stops[self.editing_color][1])
        color[self.dragging_slider] = val
        
        pos = self.stops[self.editing_color][0]
        self.stops[self.editing_color] = (pos, tuple(color))
        self._sync_hsv()
    
    def draw(self, screen):
        """
        Draw the gradient editor popup.
        
        Args:
            screen: Pygame surface to draw on
        """
        if not self.visible:
            return
        
        self._init()
        
        # Background panel
        pygame.draw.rect(screen, (45, 45, 45), (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (100, 100, 100), (self.x, self.y, self.width, self.height), 2)
        
        # Title and instructions
        title = self.font.render('Custom Gradient Editor', True, (220, 220, 220))
        screen.blit(title, (self.x + 15, self.y + 10))
        
        instructions = self.font.render(
            'Click stops to edit. Drag to move. Delete to remove.',
            True, (150, 150, 150)
        )
        screen.blit(instructions, (self.x + 15, self.y + 32))
        
        # Gradient preview bar
        self._draw_gradient_bar(screen)
        
        # Color picker (if editing a stop)
        if 0 <= self.editing_color < len(self.stops):
            self._draw_picker(screen)
        
        # Buttons
        self._draw_buttons(screen)
    
    def _draw_gradient_bar(self, screen):
        """Draw the gradient preview bar and stops."""
        bx, by, bw, bh = self._bar_rect()
        
        # Draw gradient preview
        colormap = self.get_colormap()
        for px in range(bw):
            color_idx = int(px / bw * (NUM_COLORS - 1))
            color = tuple(colormap[color_idx])
            pygame.draw.line(screen, color, (bx + px, by), (bx + px, by + bh))
        
        pygame.draw.rect(screen, (100, 100, 100), (bx, by, bw, bh), 1)
        
        # Draw color stops as triangular markers
        for i, (pos, color) in enumerate(self.stops):
            stop_x = int(bx + pos * bw)
            
            # Triangle pointing up from below the bar
            points = [
                (stop_x, by + bh),
                (stop_x - 8, by + bh + 15),
                (stop_x + 8, by + bh + 15)
            ]
            
            # Fill color - highlight selected
            fill_color = (255, 255, 100) if i == self.selected_stop else color
            pygame.draw.polygon(screen, fill_color, points)
            
            # Border - thicker for selected
            border_color = (255, 255, 255) if i == self.selected_stop else (200, 200, 200)
            border_width = 2 if i == self.selected_stop else 1
            pygame.draw.polygon(screen, border_color, points, border_width)
    
    def _draw_picker(self, screen):
        """Draw the color picker controls."""
        px, py, ps, _ = self._picker_rect()
        color = self.stops[self.editing_color][1]
        
        # Position label
        pos_text = f'Position: {self.stops[self.editing_color][0]:.2f}'
        screen.blit(self.font.render(pos_text, True, (180, 180, 180)), (px, py - 20))
        
        # Saturation/Value square
        if self.sv_surface:
            screen.blit(self.sv_surface, (px, py))
        pygame.draw.rect(screen, (80, 80, 80), (px, py, ps, ps), 1)
        
        # Color position indicator (circle in S/V square)
        cx = int(px + self.hsv[1] * ps)
        cy = int(py + (1 - self.hsv[2]) * ps)
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 6, 2)
        pygame.draw.circle(screen, (0, 0, 0), (cx, cy), 5, 1)
        
        # Hue bar
        hue_x = px + ps + 10
        if self.hue_surface:
            screen.blit(self.hue_surface, (hue_x, py))
        pygame.draw.rect(screen, (80, 80, 80), (hue_x, py, 20, 150), 1)
        
        # Hue indicator
        hy = int(py + self.hsv[0] * 150)
        pygame.draw.rect(screen, (255, 255, 255), (hue_x - 2, hy - 2, 24, 4))
        pygame.draw.rect(screen, (0, 0, 0), (hue_x - 2, hy - 2, 24, 4), 1)
        
        # Color preview and RGB values
        slider_x = self.x + 210
        slider_y = self.y + 160
        
        pygame.draw.rect(screen, color, (slider_x, py - 20, 50, 30))
        pygame.draw.rect(screen, (100, 100, 100), (slider_x, py - 20, 50, 30), 1)
        
        rgb_text = f'R:{color[0]} G:{color[1]} B:{color[2]}'
        screen.blit(self.font.render(rgb_text, True, (180, 180, 180)), (slider_x + 55, py - 12))
        
        # RGB sliders
        for i, (label, value) in enumerate(zip('RGB', color)):
            sy = slider_y + i * 35
            
            # Label
            screen.blit(self.font.render(f'{label}:', True, (180, 180, 180)), (slider_x - 20, sy + 2))
            
            # Gradient showing color component range
            for sx in range(180):
                preview_color = list(color)
                preview_color[i] = int(sx / 180 * 255)
                pygame.draw.line(screen, preview_color, (slider_x + sx, sy), (slider_x + sx, sy + 18))
            
            pygame.draw.rect(screen, (80, 80, 80), (slider_x, sy, 180, 18), 1)
            
            # Slider thumb
            thumb_x = int(slider_x + value / 255 * 180)
            pygame.draw.rect(screen, (220, 220, 220), (thumb_x - 4, sy - 2, 8, 22))
            pygame.draw.rect(screen, (100, 100, 100), (thumb_x - 4, sy - 2, 8, 22), 1)
            
            # Value text
            screen.blit(self.font.render(str(value), True, (220, 220, 220)), (slider_x + 190, sy + 2))
    
    def _draw_buttons(self, screen):
        """Draw the Add/Cancel/Apply buttons."""
        buttons = [
            ((self.x + 20, self.y + self.height - 40, 80, 28), 'Add Stop', (60, 80, 60)),
            ((self.x + self.width - 150, self.y + self.height - 40, 60, 28), 'Cancel', (80, 60, 60)),
            ((self.x + self.width - 80, self.y + self.height - 40, 60, 28), 'Apply', (60, 80, 100)),
        ]
        
        for rect, text, bg_color in buttons:
            pygame.draw.rect(screen, bg_color, rect)
            border_color = tuple(c + 40 for c in bg_color)
            pygame.draw.rect(screen, border_color, rect, 1)
            
            text_surface = self.font.render(text, True, (220, 220, 220))
            text_x = rect[0] + (rect[2] - text_surface.get_width()) // 2
            text_y = rect[1] + 6
            screen.blit(text_surface, (text_x, text_y))
