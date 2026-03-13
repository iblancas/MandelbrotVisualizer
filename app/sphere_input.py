"""Input controller for interactive Riemann sphere rotation."""


class SphereInputController:
    """Tracks drag-based yaw/pitch orientation for sphere rendering."""

    PITCH_LIMIT = 1.45

    def __init__(self, viz_width, viz_height, drag_sensitivity=0.006):
        self.width = viz_width
        self.height = viz_height
        self.drag_sensitivity = drag_sensitivity

        self.yaw = 0.0
        self.pitch = 0.0

        self.dragging = False
        self.last_pos = None

    def in_viz_area(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def handle_mouse_down(self, event):
        if event.button == 1 and self.in_viz_area(event.pos):
            self.dragging = True
            self.last_pos = event.pos
            return True
        return False

    def handle_mouse_up(self, event):
        if event.button == 1:
            was_dragging = self.dragging
            self.dragging = False
            self.last_pos = None
            return was_dragging
        return False

    def handle_mouse_motion(self, event):
        if not self.dragging or self.last_pos is None:
            return False

        mx, my = event.pos
        lx, ly = self.last_pos
        dx = mx - lx
        dy = my - ly

        self.yaw += dx * self.drag_sensitivity
        self.pitch += dy * self.drag_sensitivity

        if self.pitch > self.PITCH_LIMIT:
            self.pitch = self.PITCH_LIMIT
        elif self.pitch < -self.PITCH_LIMIT:
            self.pitch = -self.PITCH_LIMIT

        self.last_pos = event.pos
        return abs(dx) > 0 or abs(dy) > 0

    def reset(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.dragging = False
        self.last_pos = None
