"""
Menu subpackage for the Mandelbrot Visualizer.

This package provides the settings menu UI components:

- Menu: Main settings panel with dropdowns and controls
- TextInput: Text input field widget
- Slider: Float value slider widget
- Dropdown: Expandable selection widget
- GradientEditor: Custom colormap editor popup
- parse_formula: Formula string parser utility

Typical usage:
    from menu import Menu
    
    menu = Menu(x=10, y=10, screen_width=800, screen_height=800)
    handled, recompute = menu.handle_event(event)
    menu.draw(screen)
"""

from menu.base import Menu, parse_formula
from menu.widgets import TextInput, Slider, Dropdown
from menu.gradient_editor import GradientEditor

__all__ = [
    'Menu',
    'parse_formula',
    'TextInput',
    'Slider',
    'Dropdown',
    'GradientEditor',
]
