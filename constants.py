"""Shared constants loaded from settings.json."""
import json
from pathlib import Path

_settings_path = Path(__file__).parent / "settings.json"
with open(_settings_path) as f:
    SETTINGS = json.load(f)

FUNCTIONS = SETTINGS["functions"]
MENU_ORDER = SETTINGS["menu_order"]
FORMULA_PATTERNS = SETTINGS["formula_patterns"]
POWER_TO_FUNC_ID = SETTINGS["power_to_func_id"]
DEFAULTS = SETTINGS["defaults"]
OPTIONS = SETTINGS["options"]

# Precompute lookup dicts for fast access
FUNC_DEGREES = {int(k): v["degree"] for k, v in FUNCTIONS.items()}
FUNC_TRANSCENDENTAL = {int(k): v["transcendental"] for k, v in FUNCTIONS.items()}
FUNC_NAMES = {int(k): v["name"] for k, v in FUNCTIONS.items()}

def get_function_degree(func_id: int) -> int:
    """Return the degree of the given function."""
    return FUNC_DEGREES.get(func_id, 2)

def is_transcendental(func_id: int) -> bool:
    """Return True if the function is transcendental."""
    return FUNC_TRANSCENDENTAL.get(func_id, False)

def get_function_name(func_id: int) -> str:
    """Return the display name of the given function."""
    return FUNC_NAMES.get(func_id, "z^2 + c")

def get_menu_items() -> list:
    """Return list of (func_id, name) for menu display."""
    return [(fid, FUNC_NAMES[fid]) for fid in MENU_ORDER]
