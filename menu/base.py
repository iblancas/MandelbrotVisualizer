"""
Main Menu class for the settings panel.

This module provides the primary menu interface for configuring the
Mandelbrot/Julia set visualization. Features include:

- Iteration count selection
- Function selection (z^n + c, transcendental functions)
- Custom formula input
- Escape radius configuration
- Colormap selection with custom gradient support
- GPU/CPU toggle
- Julia/Mandelbrot mode switch with c parameter sliders
- High-resolution image export

The menu uses dropdown widgets for selections and integrates with
the gradient editor for custom color schemes.
"""

import re
import pygame

from colormaps import COLORMAPS
from constants import (
    FUNC_NAMES, MENU_ORDER, FORMULA_PATTERNS, POWER_TO_FUNC_ID, DEFAULTS, OPTIONS
)
from custom_formula import validate_custom_formula
from menu.widgets import TextInput, Slider, Dropdown
from menu.gradient_editor import GradientEditor
from sphere.modes import (
    PLANE_MODE,
    SPHERE_PARAMETER_MODE,
    SPHERE_JULIA_MODE,
    DOMAIN_MODE_LABELS,
    DOMAIN_LABEL_TO_MODE,
)


def parse_formula(formula):
    """
    Parse a user-entered formula and return the corresponding function ID.
    
    This function handles various input formats and normalizes them:
    - Unicode superscripts (², ³, ⁴, etc.) → ** notation
    - Caret notation (^) → ** notation  
    - Multiplication symbols (·, ×) → *
    - Complex conjugate notation (z_bar, z̄) → conj(z)
    - Common transcendental patterns
    
    Supported formulas match entries in FORMULA_PATTERNS or follow
    the pattern z^n + c for n from 2 to 8.
    
    Args:
        formula: User-entered formula string
        
    Returns:
        Tuple of (func_id, display_name) on success
        Tuple of (None, error_message) on failure
        
    Examples:
        >>> parse_formula("z^2 + c")
        (0, 'z² + c')
        >>> parse_formula("z³+c")
        (1, 'z³ + c')
        >>> parse_formula("sin(z) * c")
        (8, 'sin(z)·c')
    """
    # Normalize the formula
    f = formula.lower().replace(' ', '').replace('·', '*').replace('×', '*')
    f = f.replace('^', '**').replace('²', '**2').replace('³', '**3')
    
    # Handle Unicode superscripts 4-8
    for i, sup in enumerate('⁴⁵⁶⁷⁸', 4):
        f = f.replace(sup, f'**{i}')
    
    # Handle complex conjugate notation
    f = f.replace('z_bar', 'conj(z)').replace('z̄', 'conj(z)')
    
    # Handle transcendental functions written as f(z)+c → c*f(z)
    f = f.replace('e**z', 'exp(z)')
    f = f.replace('sin(z)+c', 'c*sin(z)')
    f = f.replace('cos(z)+c', 'c*cos(z)')
    f = f.replace('exp(z)+c', 'c*exp(z)')
    
    # Check against known patterns
    if f in FORMULA_PATTERNS:
        func_id = FORMULA_PATTERNS[f]
        return func_id, FUNC_NAMES.get(func_id, 'z^2 + c')
    
    # Check for z^n + c pattern
    match = re.match(r'^z\*\*(\d+)\+c$', f)
    if match:
        n = match.group(1)
        if n in POWER_TO_FUNC_ID:
            return int(POWER_TO_FUNC_ID[n]), f'z^{n} + c'
        return None, f'z^{n} not supported (max 8)'
    
    return None, 'Unrecognized formula'


class Menu:
    """
    Settings menu with dropdowns, sliders, and gradient editor.
    
    The menu provides a collapsible panel for adjusting visualization
    parameters. When expanded, it shows:
    
    1. Max Iterations dropdown (affects detail and render time)
    2. Function selector dropdown with custom formula input
    3. Escape radius dropdown (bailout threshold)
    4. Colormap dropdown with preview and custom gradient option
    5. GPU toggle button (when GPU available)
    6. Julia/Mandelbrot mode toggle with c parameter sliders
    7. Save High-Res Image button
    
    State Management:
    - save_requested: Set True when save button clicked
    - gpu_toggle_requested: Set True when GPU toggle clicked
    - using_custom: True when using custom colormap
    - julia_mode: True when displaying Julia set instead of Mandelbrot
    
    Attributes:
        expanded: Whether the menu panel is open
        max_iter: Current maximum iteration count
        colormap_name: Name of selected colormap
        func_id: ID of selected iteration function
        func_display: Display name of current function
        escape_radius: Bailout radius for iteration
        use_gpu: Preference for GPU computation
        julia_mode: Whether to display Julia set
        julia_c_real: Real part of Julia c parameter
        julia_c_imag: Imaginary part of Julia c parameter
    """
    
    def __init__(self, x, y, width=220, screen_width=800, screen_height=800):
        """
        Initialize the settings menu.
        
        The menu is always visible in the sidebar - there is no toggle.
        
        Args:
            x: X position of the menu
            y: Y position of the menu
            width: Width of the menu panel
            screen_width: Screen width for centering gradient editor
            screen_height: Screen height for centering gradient editor
        """
        self.x, self.y, self.width = x, y, width
        self.height = screen_height  # Fill the sidebar
        self.expanded = True  # Always expanded in sidebar mode
        self.font = None
        
        # Visualization parameters (loaded from defaults)
        self.max_iter = DEFAULTS['max_iter']
        self.colormap_name = DEFAULTS['colormap']
        self.func_id = DEFAULTS['func_id']
        self.escape_radius = DEFAULTS['escape_radius']
        
        # Build function list from menu order
        self.funcs = [(FUNC_NAMES[fid], fid) for fid in MENU_ORDER]
        self.colormap_names = list(COLORMAPS.keys()) + ['Custom...']
        self.func_display = FUNC_NAMES.get(self.func_id, 'z^2 + c')
        
        # Custom formula/colormap state
        self.custom_formula = None
        self.custom_colormap = None
        self.using_custom = False
        
        # GPU state
        self.use_gpu = True
        self.gpu_available = False
        self.gpu_device_name = "Checking..."
        self.gpu_toggle_rect = None
        self.save_requested = False
        self.gpu_toggle_requested = False
        
        # Dropdown widgets (initialized lazily)
        self.dropdowns = {}
        self.func_input = None
        self.formula_error = None
        self.save_button_rect = None
        
        # Gradient editor popup
        self.gradient_editor = GradientEditor(screen_width, screen_height)
        
        # Julia set mode
        julia_c_default = DEFAULTS.get('julia_c', [-0.7, 0.27015])
        self.julia_mode = False
        self.julia_c_real = julia_c_default[0]
        self.julia_c_imag = julia_c_default[1]
        self.julia_c_abs_bound = 2.0
        self.domain_mode = PLANE_MODE
        self.domain_toggle_rect = None
        self.julia_toggle_rect = None
        self.julia_sliders = {}
        self.julia_range_slider = None
    
    def _init(self):
        """Initialize font on first draw."""
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 12)
    
    def _init_dropdowns(self):
        """
        Initialize dropdown widgets when menu first expands.
        
        Creates dropdowns for iterations, function, escape radius,
        and colormap selection, plus the formula text input and
        Julia parameter sliders.
        """
        w = self.width - 16
        iter_opts = OPTIONS['iterations']
        esc_opts = OPTIONS['escape_radius']
        
        # Iteration count dropdown
        self.dropdowns['iter'] = Dropdown(
            self.x + 8, 0, w,
            [str(v) for v in iter_opts],
            iter_opts.index(self.max_iter) if self.max_iter in iter_opts else 0
        )

        # Domain mode dropdown
        domain_options = list(DOMAIN_MODE_LABELS.values())
        self.dropdowns['domain'] = Dropdown(
            self.x + 8, 0, w,
            domain_options,
            domain_options.index(DOMAIN_MODE_LABELS[self.domain_mode]),
        )
        
        # Function selection dropdown
        self.dropdowns['func'] = Dropdown(
            self.x + 8, 0, w,
            [n for n, _ in self.funcs],
            next((i for i, (_, fid) in enumerate(self.funcs) if fid == self.func_id), 0)
        )
        
        # Escape radius dropdown
        self.dropdowns['escape'] = Dropdown(
            self.x + 8, 0, w,
            [str(v) for v in esc_opts],
            esc_opts.index(int(self.escape_radius)) if int(self.escape_radius) in esc_opts else 0
        )
        
        # Colormap dropdown
        self.dropdowns['color'] = Dropdown(
            self.x + 8, 0, w,
            self.colormap_names,
            self.colormap_names.index(self.colormap_name) if self.colormap_name in self.colormap_names else 0
        )
        
        # Custom formula input
        self.func_input = TextInput(self.x + 8, 0, w, 24, "Type formula: z^3+c")
        
        # Julia c parameter sliders
        self.julia_sliders = {
            'c_real': Slider(self.x + 8, 0, w, -self.julia_c_abs_bound, self.julia_c_abs_bound, self.julia_c_real, "c real"),
            'c_imag': Slider(self.x + 8, 0, w, -self.julia_c_abs_bound, self.julia_c_abs_bound, self.julia_c_imag, "c imag"),
        }
        self.julia_range_slider = Slider(self.x + 8, 0, w, 0.25, 20.0, self.julia_c_abs_bound, "|c| bound")
        self._set_julia_c_bound(self.julia_c_abs_bound)
    
    def get_rect(self):
        """
        Get the bounding rectangle of the menu.
        
        Returns:
            pygame.Rect of the menu area
        """
        # Menu fills the sidebar
        return pygame.Rect(self.x, self.y, self.width, self.height - 20)
    
    def get_colormap(self):
        """
        Get the current colormap as a numpy array.
        
        Returns:
            NumPy array of shape (256, 3) with RGB values
        """
        if self.using_custom and self.custom_colormap is not None:
            return self.custom_colormap
        return COLORMAPS[self.colormap_name]()
    
    def handle_event(self, event):
        """
        Handle pygame events for the menu.
        
        Args:
            event: Pygame event to process
            
        Returns:
            Tuple of (handled: bool, need_recompute: bool)
        """
        # Gradient editor takes priority
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
        
        # Route to appropriate handler
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self._handle_click(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            return self._handle_mouse_up(event)
        elif event.type == pygame.KEYDOWN:
            return self._handle_key(event)
        elif event.type == pygame.MOUSEMOTION:
            return self._handle_motion(event)
        
        return False, False
    
    def _handle_click(self, pos):
        """
        Handle mouse click events.

        Args:
            pos: (x, y) mouse position

        Returns:
            Tuple of (handled, need_recompute)
        """
        mx, my = pos

        # Check if click is in menu area
        if not self.get_rect().collidepoint(mx, my):
            return False, False

        # Check dropdowns
        handlers = [
            ('domain', self._on_domain_change),
            ('iter', lambda v: setattr(self, 'max_iter', int(v))),
            ('func', self._on_func_change),
            ('escape', lambda v: setattr(self, 'escape_radius', float(v))),
            ('color', self._on_color_change),
        ]

        for name, on_change in handlers:
            dd = self.dropdowns.get(name)
            if dd:
                handled, changed = dd.handle_event(
                    pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1)
                )
                if handled:
                    if changed:
                        on_change(dd.value)
                    return True, changed
        
        # Formula input
        if self.func_input:
            handled, _ = self.func_input.handle_event(
                pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1)
            )
            if handled:
                return True, False
        
        # Save button
        if self.save_button_rect and self.save_button_rect.collidepoint(mx, my):
            self.save_requested = True
            return True, False
        
        # GPU toggle
        if self.gpu_toggle_rect and self.gpu_toggle_rect.collidepoint(mx, my):
            if self.gpu_available:
                self.use_gpu = not self.use_gpu
                self.gpu_toggle_requested = True
            return True, False
        
        # Julia mode toggle
        if self.julia_toggle_rect and self.julia_toggle_rect.collidepoint(mx, my):
            self.julia_mode = not self.julia_mode
            return True, True  # Recompute when mode changes
        
        # Julia sliders (only in Julia mode)
        if self._show_julia_controls():
            if self.julia_range_slider is not None:
                handled, changed = self.julia_range_slider.handle_event(
                    pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1)
                )
                if handled:
                    changed_c = False
                    if changed:
                        changed_c = self._set_julia_c_bound(self.julia_range_slider.value)
                    return True, changed_c

            for slider in self.julia_sliders.values():
                handled, changed = slider.handle_event(
                    pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1)
                )
                if handled:
                    if changed:
                        self.julia_c_real = self.julia_sliders['c_real'].value
                        self.julia_c_imag = self.julia_sliders['c_imag'].value
                    return True, changed
        
        return self.get_rect().collidepoint(mx, my), False
    
    def _handle_mouse_up(self, event):
        """Handle mouse button release for slider interactions."""
        if self._show_julia_controls():
            if self.julia_range_slider is not None:
                handled, _ = self.julia_range_slider.handle_event(event)
                if handled:
                    return True, False
            for slider in self.julia_sliders.values():
                handled, _ = slider.handle_event(event)
                if handled:
                    return True, False
        # Only claim to handle if mouse is actually in menu area
        return self.get_rect().collidepoint(event.pos), False
    
    def _handle_motion(self, event):
        """Handle mouse motion for dropdown hover and slider drag."""
        # Update dropdown hover states
        for dd in self.dropdowns.values():
            dd.handle_event(event)
        
        # Julia slider dragging
        if self._show_julia_controls():
            if self.julia_range_slider is not None:
                handled, changed = self.julia_range_slider.handle_event(event)
                if handled:
                    changed_c = False
                    if changed:
                        changed_c = self._set_julia_c_bound(self.julia_range_slider.value)
                    return True, changed_c

            for slider in self.julia_sliders.values():
                handled, changed = slider.handle_event(event)
                if handled and changed:
                    self.julia_c_real = self.julia_sliders['c_real'].value
                    self.julia_c_imag = self.julia_sliders['c_imag'].value
                    return True, True  # Recompute on slider drag
        
        return False, False
    
    def _on_func_change(self, value):
        """Handle function dropdown selection change."""
        for name, fid in self.funcs:
            if name == value:
                self.func_id = fid
                self.func_display = name
                self.custom_formula = None
                self.formula_error = None
                if self.func_input:
                    self.func_input.text = ""
                break
    
    def _on_color_change(self, value):
        """Handle colormap dropdown selection change."""
        if value == 'Custom...':
            self.gradient_editor.show()
        else:
            self.colormap_name = value
            self.using_custom = False

    def _on_domain_change(self, value):
        """Handle domain mode dropdown selection change."""
        self.domain_mode = DOMAIN_LABEL_TO_MODE.get(value, PLANE_MODE)

    def _show_julia_controls(self):
        """Return True when Julia c sliders should be visible."""
        return self.domain_mode == SPHERE_JULIA_MODE or self.julia_mode

    def _set_julia_c_bound(self, abs_bound):
        """Update symmetric Julia c slider bounds and clamp c values if needed."""
        bound = max(0.01, float(abs_bound))
        self.julia_c_abs_bound = bound

        if self.julia_range_slider is not None:
            self.julia_range_slider.value = bound

        if not self.julia_sliders:
            return False

        real_slider = self.julia_sliders['c_real']
        imag_slider = self.julia_sliders['c_imag']
        real_slider.min_val = -bound
        real_slider.max_val = bound
        imag_slider.min_val = -bound
        imag_slider.max_val = bound

        old_real = self.julia_c_real
        old_imag = self.julia_c_imag
        self.julia_c_real = max(-bound, min(bound, self.julia_c_real))
        self.julia_c_imag = max(-bound, min(bound, self.julia_c_imag))
        real_slider.value = self.julia_c_real
        imag_slider.value = self.julia_c_imag

        return (
            abs(self.julia_c_real - old_real) > 1e-9
            or abs(self.julia_c_imag - old_imag) > 1e-9
        )
    
    def _handle_key(self, event):
        """
        Handle keyboard events for formula input.
        
        Args:
            event: Pygame KEYDOWN event
            
        Returns:
            Tuple of (handled, need_recompute)
        """
        if not self.func_input or not self.func_input.active:
            return False, False
        
        handled, changed = self.func_input.handle_event(event)
        
        # Clear error when user starts typing
        if changed and self.formula_error:
            self.formula_error = None
        
        # Enter key submits formula
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.func_input.text.strip():
                formula_text = self.func_input.text.strip()
                result, msg = parse_formula(formula_text)
                if result is not None:
                    self.func_id = result
                    self.func_display = msg
                    self.custom_formula = None
                else:
                    _, error = validate_custom_formula(formula_text)
                    if error:
                        self.formula_error = error
                        return True, False

                    self.custom_formula = formula_text
                    self.func_display = formula_text
                self.formula_error = None
                return True, True
        
        return handled, False
    
    def draw(self, screen):
        """
        Draw the menu to the screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        self._init()
        
        # Initialize dropdowns if not done yet
        if not self.dropdowns:
            self._init_dropdowns()
        
        # Draw menu content (always visible in sidebar)
        self._draw_sidebar(screen)
        
        # Gradient editor overlay
        self.gradient_editor.draw(screen)
    
    def _draw_sidebar(self, screen):
        """Draw the sidebar menu with all controls."""
        # Title
        title_font = pygame.font.SysFont('Arial', 16, bold=True)
        title = title_font.render('Settings', True, (220, 220, 220))
        screen.blit(title, (self.x + 8, self.y + 5))
        
        y = self.y + 35
        
        # Draw each section: label + dropdown + optional extras
        sections = [
            ('Domain:', 'domain', None),
            ('Max Iterations:', 'iter', None),
            ('Function f(z):', 'func', 'input'),
            ('Escape Radius:', 'escape', None),
            ('Color Scheme:', 'color', 'preview'),
        ]
        
        for label, key, extra in sections:
            # Section label
            screen.blit(
                self.font.render(label, True, (180, 180, 180)),
                (self.x + 8, y)
            )
            y += 18
            
            # Dropdown
            dd = self.dropdowns.get(key)
            if dd:
                dd.y = y
                dd.draw(screen, self.font)
                y += dd.height
                if dd.expanded:
                    y += len(dd.options) * 22
            
            # Extra: formula input
            if extra == 'input' and self.func_input:
                y += 4
                screen.blit(
                    self.font.render('or type formula:', True, (140, 140, 140)),
                    (self.x + 8, y)
                )
                y += 16
                self.func_input.x, self.func_input.y = self.x + 8, y
                self.func_input.draw(screen, self.font)
                y += self.func_input.height
                if self.formula_error:
                    screen.blit(
                        self.font.render(self.formula_error, True, (255, 100, 100)),
                        (self.x + 8, y + 2)
                    )
                    y += 14
            
            # Extra: colormap preview
            if extra == 'preview' and dd and not dd.expanded:
                y += 5
                self._draw_colormap_preview(screen, y)
                y += 18
            
            y += 10
        
        # GPU section
        y = self._draw_gpu_section(screen, y)
        
        # Julia mode section
        y = self._draw_julia_section(screen, y)
        
        # Save button
        self._draw_save_button(screen, y)
    
    def _draw_colormap_preview(self, screen, y):
        """Draw a preview bar of the current colormap."""
        cmap = self.get_colormap()
        pw = self.width - 16
        for px in range(pw):
            idx = int(px / pw * (len(cmap) - 1))
            pygame.draw.line(
                screen, tuple(cmap[idx]),
                (self.x + 8 + px, y),
                (self.x + 8 + px, y + 12)
            )
    
    def _draw_gpu_section(self, screen, y):
        """Draw the GPU toggle section."""
        screen.blit(
            self.font.render('GPU Acceleration:', True, (180, 180, 180)),
            (self.x + 8, y)
        )
        y += 18
        
        self.gpu_toggle_rect = pygame.Rect(self.x + 8, y, self.width - 16, 26)
        
        if self.gpu_available:
            if self.use_gpu:
                bg, border, tc = (70, 120, 70), (100, 180, 100), (220, 255, 220)
                label = "GPU ON"
            else:
                bg, border, tc = (100, 70, 70), (150, 100, 100), (255, 220, 220)
                label = "GPU OFF (CPU)"
        else:
            bg, border, tc = (60, 60, 60), (100, 100, 100), (150, 150, 150)
            label = "GPU N/A"
        
        pygame.draw.rect(screen, bg, self.gpu_toggle_rect)
        pygame.draw.rect(screen, border, self.gpu_toggle_rect, 1)
        
        t = self.font.render(label, True, tc)
        screen.blit(t, (
            self.gpu_toggle_rect.centerx - t.get_width() // 2,
            self.gpu_toggle_rect.centery - t.get_height() // 2
        ))
        y += 28
        
        # Device name
        screen.blit(
            self.font.render(self.gpu_device_name, True, (120, 120, 120)),
            (self.x + 8, y)
        )
        y += 22
        
        return y
    
    def _draw_julia_section(self, screen, y):
        """Draw the Julia/Mandelbrot mode toggle section."""
        if self.domain_mode == SPHERE_PARAMETER_MODE:
            screen.blit(
                self.font.render('Sphere mode: parameter plane on Riemann sphere', True, (140, 140, 140)),
                (self.x + 8, y)
            )
            y += 18
            screen.blit(
                self.font.render('Drag: rotate sphere | Wheel: zoom', True, (120, 120, 120)),
                (self.x + 8, y)
            )
            y += 20
            return y

        if self.domain_mode == SPHERE_JULIA_MODE:
            screen.blit(
                self.font.render('Sphere Julia mode (c fixed):', True, (180, 180, 180)),
                (self.x + 8, y)
            )
            y += 20
            screen.blit(
                self.font.render('Drag: rotate sphere | Wheel: zoom', True, (120, 120, 120)),
                (self.x + 8, y)
            )
            y += 20

            screen.blit(
                self.font.render('Julia c parameter:', True, (140, 140, 140)),
                (self.x + 8, y)
            )
            y += 18

            if self.julia_range_slider is not None:
                self.julia_range_slider.x, self.julia_range_slider.y = self.x + 8, y + 14
                self.julia_range_slider.draw(screen, self.font)
                y += 43

            for key in ['c_real', 'c_imag']:
                slider = self.julia_sliders.get(key)
                if slider:
                    slider.x, slider.y = self.x + 8, y + 14
                    slider.draw(screen, self.font)
                    y += 43

            return y

        screen.blit(
            self.font.render('Visualization Mode:', True, (180, 180, 180)),
            (self.x + 8, y)
        )
        y += 18
        
        self.julia_toggle_rect = pygame.Rect(self.x + 8, y, self.width - 16, 26)
        
        if self.julia_mode:
            bg, border, tc = (100, 70, 120), (150, 100, 180), (255, 220, 255)
            label = "Julia Set (c fixed)"
        else:
            bg, border, tc = (70, 100, 120), (100, 150, 180), (220, 255, 255)
            label = "Mandelbrot (z=0)"
        
        pygame.draw.rect(screen, bg, self.julia_toggle_rect)
        pygame.draw.rect(screen, border, self.julia_toggle_rect, 1)
        
        t = self.font.render(label, True, tc)
        screen.blit(t, (
            self.julia_toggle_rect.centerx - t.get_width() // 2,
            self.julia_toggle_rect.centery - t.get_height() // 2
        ))
        y += 35
        
        # Julia c parameter sliders (only in Julia mode)
        if self.julia_mode:
            screen.blit(
                self.font.render('Julia c parameter:', True, (140, 140, 140)),
                (self.x + 8, y)
            )
            y += 18

            if self.julia_range_slider is not None:
                self.julia_range_slider.x, self.julia_range_slider.y = self.x + 8, y + 14
                self.julia_range_slider.draw(screen, self.font)
                y += 43
            
            for key in ['c_real', 'c_imag']:
                slider = self.julia_sliders.get(key)
                if slider:
                    slider.x, slider.y = self.x + 8, y + 14
                    slider.draw(screen, self.font)
                    y += 43
        
        return y
    
    def _draw_save_button(self, screen, y):
        """Draw the save image button."""
        self.save_button_rect = pygame.Rect(self.x + 8, y, self.width - 16, 26)
        pygame.draw.rect(screen, (70, 100, 70), self.save_button_rect)
        pygame.draw.rect(screen, (100, 150, 100), self.save_button_rect, 1)
        
        t = self.font.render('Save High-Res Image', True, (220, 255, 220))
        screen.blit(t, (
            self.save_button_rect.centerx - t.get_width() // 2,
            self.save_button_rect.centery - t.get_height() // 2
        ))
    
    def point_in_menu(self, pos):
        """
        Check if a point is within the menu area.
        
        Args:
            pos: (x, y) position to check
            
        Returns:
            True if the point is within the menu or gradient editor
        """
        if self.gradient_editor.visible:
            ge_rect = pygame.Rect(
                self.gradient_editor.x, self.gradient_editor.y,
                self.gradient_editor.width, self.gradient_editor.height
            )
            if ge_rect.collidepoint(pos):
                return True
        return self.get_rect().collidepoint(pos)
    
    def update_gpu_status(self, available, enabled, device_name):
        """
        Update GPU availability and status display.
        
        Args:
            available: Whether GPU is available
            enabled: Whether GPU is currently enabled
            device_name: Name of the GPU device
        """
        self.gpu_available = available
        self.use_gpu = enabled
        
        # Truncate long device names
        if len(device_name) > 28:
            self.gpu_device_name = device_name[:25] + "..."
        else:
            self.gpu_device_name = device_name
