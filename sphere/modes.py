"""Domain mode constants for plane and Riemann sphere visualizations."""

PLANE_MODE = 'plane'
SPHERE_PARAMETER_MODE = 'sphere_parameter'
SPHERE_JULIA_MODE = 'sphere_julia'

DOMAIN_MODE_LABELS = {
    PLANE_MODE: 'Plane (Standard)',
    SPHERE_PARAMETER_MODE: 'Sphere (Parameter)',
    SPHERE_JULIA_MODE: 'Sphere (Julia)',
}

DOMAIN_LABEL_TO_MODE = {label: mode for mode, label in DOMAIN_MODE_LABELS.items()}


def is_sphere_mode(mode):
    """Return True when mode uses Riemann sphere rendering."""
    return mode in (SPHERE_PARAMETER_MODE, SPHERE_JULIA_MODE)
