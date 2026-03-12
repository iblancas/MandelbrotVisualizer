"""
UI widget components for the Mandelbrot visualizer menu.

This module contains reusable UI components used throughout the settings
menu, including text inputs, sliders, and dropdown selectors.

All widgets follow a consistent API:
- handle_event(event) -> (handled: bool, changed: bool)
- draw(screen, font) -> None

Widgets are designed to be self-contained and manage their own state
for interactions like dragging, text editing, and selection.
"""

import pygame


class TextInput:
    """
    Text input field component with cursor support.
    
    Supports standard text editing features:
    - Cursor positioning with arrow keys, Home, End
    - Backspace and Delete for character removal
    - Enter/Return to submit
    - Visual feedback for active/inactive states
    
    Attributes:
        x, y: Position of the input field
        width, height: Dimensions of the input field
        text: Current text content
        placeholder: Gray text shown when empty
        active: Whether the field is focused for input
        cursor_pos: Current cursor position in text
    """
    
    def __init__(self, x, y, width, height=24, placeholder=""):
        """
        Initialize a text input field.
        
        Args:
            x: X position of the input
            y: Y position of the input
            width: Width of the input field
            height: Height of the input field (default 24)
            placeholder: Placeholder text when empty
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = ""
        self.placeholder = placeholder
        self.active = False
        self.cursor_pos = 0
        self._cursor_timer = 0  # For cursor blink animation
    
    @property
    def rect(self):
        """Get the bounding rectangle of this input."""
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def handle_event(self, event):
        """
        Handle pygame events for this text input.
        
        Processes mouse clicks (activate/deactivate) and keyboard
        input when active.
        
        Args:
            event: Pygame event to process
            
        Returns:
            Tuple of (handled: bool, text_changed: bool)
        """
        # Click handling - activate if clicked, deactivate otherwise
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            was_active = self.active
            self.active = self.rect.collidepoint(event.pos)
            return self.active or was_active, False
        
        # Keyboard handling when active
        if event.type == pygame.KEYDOWN and self.active:
            old_text = self.text
            key = event.key
            
            if key == pygame.K_BACKSPACE and self.cursor_pos > 0:
                # Delete character before cursor
                self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                self.cursor_pos -= 1
                
            elif key == pygame.K_DELETE and self.cursor_pos < len(self.text):
                # Delete character after cursor
                self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]
                
            elif key == pygame.K_LEFT:
                # Move cursor left
                self.cursor_pos = max(0, self.cursor_pos - 1)
                
            elif key == pygame.K_RIGHT:
                # Move cursor right
                self.cursor_pos = min(len(self.text), self.cursor_pos + 1)
                
            elif key in (pygame.K_HOME, pygame.K_END):
                # Jump to start or end
                self.cursor_pos = 0 if key == pygame.K_HOME else len(self.text)
                
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                # Submit - deactivate and report if text changed
                self.active = False
                return True, old_text != self.text
                
            elif event.unicode and event.unicode.isprintable():
                # Insert character at cursor
                self.text = self.text[:self.cursor_pos] + event.unicode + self.text[self.cursor_pos:]
                self.cursor_pos += 1
            
            return True, old_text != self.text
        
        return False, False
    
    def draw(self, screen, font):
        """
        Draw the text input field.
        
        Args:
            screen: Pygame surface to draw on
            font: Pygame font for text rendering
        """
        rect = self.rect
        
        # Background - slightly brighter when active
        bg_color = (60, 60, 70) if self.active else (50, 50, 55)
        pygame.draw.rect(screen, bg_color, rect)
        
        # Border - blue when active for visual feedback
        border_color = (100, 140, 180) if self.active else (80, 80, 80)
        border_width = 2 if self.active else 1
        pygame.draw.rect(screen, border_color, rect, border_width)
        
        # Text content or placeholder
        display_text = self.text or self.placeholder
        text_color = (220, 220, 220) if self.text else (120, 120, 120)
        text_surface = font.render(display_text, True, text_color)
        screen.blit(text_surface, (rect.left + 6, rect.centery - text_surface.get_height() // 2))
        
        # Blinking cursor when active
        if self.active:
            self._cursor_timer = (self._cursor_timer + 1) % 60
            if self._cursor_timer < 30:  # Blink every 30 frames
                # Calculate cursor X position based on text before cursor
                if self.text:
                    cursor_text = font.render(self.text[:self.cursor_pos], True, text_color)
                    cursor_x = rect.left + 6 + cursor_text.get_width()
                else:
                    cursor_x = rect.left + 6
                pygame.draw.line(
                    screen, (220, 220, 220),
                    (cursor_x, rect.top + 4),
                    (cursor_x, rect.bottom - 4),
                    2
                )


class Slider:
    """
    Horizontal slider component for selecting float values.
    
    Used for parameters like Julia set c coordinates where
    continuous value selection is needed.
    
    Features:
    - Click anywhere on track to set value
    - Drag thumb for fine control
    - Shows current value in label
    
    Attributes:
        x, y: Position of the slider
        width: Width of the slider track
        min_val, max_val: Value range
        value: Current value
        label: Optional label displayed above
    """
    
    def __init__(self, x, y, width, min_val, max_val, value, label=""):
        """
        Initialize a slider.
        
        Args:
            x: X position
            y: Y position  
            width: Track width in pixels
            min_val: Minimum value
            max_val: Maximum value
            value: Initial value
            label: Optional label text
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = 20
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.dragging = False
    
    @property
    def rect(self):
        """Get the bounding rectangle of this slider."""
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def _val_to_pos(self, val):
        """
        Convert a value to pixel X position.
        
        Args:
            val: Value in [min_val, max_val]
            
        Returns:
            X pixel position of the thumb
        """
        t = (val - self.min_val) / (self.max_val - self.min_val)
        return self.x + int(t * (self.width - 10))
    
    def _pos_to_val(self, px):
        """
        Convert pixel X position to value.
        
        Args:
            px: X pixel position
            
        Returns:
            Value in [min_val, max_val]
        """
        t = max(0, min(1, (px - self.x) / (self.width - 10)))
        return self.min_val + t * (self.max_val - self.min_val)
    
    def handle_event(self, event):
        """
        Handle pygame events for this slider.
        
        Args:
            event: Pygame event to process
            
        Returns:
            Tuple of (handled: bool, value_changed: bool)
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                old_val = self.value
                self.value = self._pos_to_val(event.pos[0])
                return True, abs(self.value - old_val) > 1e-9
                
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True, False
                
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            old_val = self.value
            self.value = self._pos_to_val(event.pos[0])
            return True, abs(self.value - old_val) > 1e-9
        
        return False, False
    
    def draw(self, screen, font):
        """
        Draw the slider.
        
        Args:
            screen: Pygame surface to draw on
            font: Pygame font for label rendering
        """
        # Track background
        pygame.draw.rect(screen, (50, 50, 55), self.rect)
        pygame.draw.rect(screen, (80, 80, 80), self.rect, 1)
        
        # Filled portion (left of thumb)
        fill_width = self._val_to_pos(self.value) - self.x
        pygame.draw.rect(screen, (70, 100, 130), (self.x, self.y, fill_width, self.height))
        
        # Thumb
        thumb_x = self._val_to_pos(self.value)
        pygame.draw.rect(screen, (150, 150, 160), (thumb_x, self.y, 10, self.height))
        pygame.draw.rect(screen, (200, 200, 200), (thumb_x, self.y, 10, self.height), 1)
        
        # Label with current value
        if self.label:
            label_text = f"{self.label}: {self.value:.3f}"
            label_surface = font.render(label_text, True, (180, 180, 180))
            screen.blit(label_surface, (self.x, self.y - 14))


class Dropdown:
    """
    Dropdown select component for choosing from a list of options.
    
    Displays a collapsed button showing the current selection,
    which expands to show all options when clicked.
    
    Features:
    - Single-click to expand/collapse
    - Hover highlighting
    - Selected item highlighting
    - Automatic closing when clicking outside
    
    Attributes:
        x, y: Position of the dropdown
        width: Width of the dropdown
        options: List of option strings
        selected_idx: Index of currently selected option
        expanded: Whether the options list is visible
    """
    
    def __init__(self, x, y, width, options, selected_idx=0):
        """
        Initialize a dropdown.
        
        Args:
            x: X position
            y: Y position
            width: Width of the dropdown
            options: List of option strings
            selected_idx: Initially selected index (default 0)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = 24  # Height of collapsed dropdown
        self.options = options
        self.selected_idx = selected_idx
        self.expanded = False
        self.hovered_idx = -1  # For hover highlighting
    
    @property
    def value(self):
        """Get the currently selected option string."""
        return self.options[self.selected_idx]
    
    def set_value(self, value):
        """
        Set the selected option by value string.
        
        Args:
            value: Option string to select
        """
        if value in self.options:
            self.selected_idx = self.options.index(value)
    
    def handle_event(self, event):
        """
        Handle pygame events for this dropdown.
        
        Args:
            event: Pygame event to process
            
        Returns:
            Tuple of (handled: bool, value_changed: bool)
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            
            # Click on main dropdown button - toggle expansion
            if pygame.Rect(self.x, self.y, self.width, self.height).collidepoint(mx, my):
                self.expanded = not self.expanded
                return True, False
            
            # Click on expanded options list
            if self.expanded:
                item_y = self.y + self.height
                for i in range(len(self.options)):
                    if pygame.Rect(self.x, item_y, self.width, 22).collidepoint(mx, my):
                        old_idx = self.selected_idx
                        self.selected_idx = i
                        self.expanded = False
                        return True, old_idx != i
                    item_y += 22
                    
                # Clicked outside - close dropdown
                self.expanded = False
                return True, False
                
        elif event.type == pygame.MOUSEMOTION and self.expanded:
            # Update hover highlight
            self.hovered_idx = -1
            item_y = self.y + self.height
            for i in range(len(self.options)):
                if pygame.Rect(self.x, item_y, self.width, 22).collidepoint(event.pos):
                    self.hovered_idx = i
                    break
                item_y += 22
        
        return False, False
    
    def draw(self, screen, font):
        """
        Draw the dropdown.
        
        Args:
            screen: Pygame surface to draw on
            font: Pygame font for text rendering
        """
        # Main button background
        pygame.draw.rect(screen, (55, 55, 55), (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (100, 100, 100), (self.x, self.y, self.width, self.height), 1)
        
        # Selected value text
        value_surface = font.render(str(self.value), True, (220, 220, 220))
        screen.blit(value_surface, (self.x + 8, self.y + 5))
        
        # Arrow indicator
        arrow = "▲" if self.expanded else "▼"
        arrow_surface = font.render(arrow, True, (150, 150, 150))
        screen.blit(arrow_surface, (self.x + self.width - 18, self.y + 5))
        
        # Expanded options list
        if self.expanded:
            item_y = self.y + self.height
            for i, option in enumerate(self.options):
                # Background color based on state
                if i == self.selected_idx:
                    bg_color = (70, 100, 70)  # Green for selected
                elif i == self.hovered_idx:
                    bg_color = (65, 65, 65)  # Light for hovered
                else:
                    bg_color = (50, 50, 50)  # Default
                
                pygame.draw.rect(screen, bg_color, (self.x, item_y, self.width, 22))
                pygame.draw.rect(screen, (80, 80, 80), (self.x, item_y, self.width, 22), 1)
                
                # Option text
                text_color = (255, 255, 255) if i == self.selected_idx else (180, 180, 180)
                option_surface = font.render(str(option), True, text_color)
                screen.blit(option_surface, (self.x + 8, item_y + 4))
                
                item_y += 22
