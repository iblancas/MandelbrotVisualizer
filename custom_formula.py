"""Helpers for custom fractal formula parsing, validation, and evaluation."""

from dataclasses import dataclass
import ast
import cmath
import re

import numba
import numpy as np


class CustomFormulaError(ValueError):
    """Raised when a custom formula cannot be prepared or validated."""


@dataclass(frozen=True)
class PreparedCustomFormula:
    """Compiled representation of a user-defined formula."""

    original: str
    normalized: str
    compiled_code: object
    jit_iter_func: object = None


SAFE_EVAL_GLOBALS = {
    '__builtins__': {},
    'sin': cmath.sin,
    'cos': cmath.cos,
    'tan': cmath.tan,
    'exp': cmath.exp,
    'log': cmath.log,
    'sqrt': cmath.sqrt,
    'abs': abs,
    'conj': lambda x: complex(x.real, -x.imag),
    'sinh': cmath.sinh,
    'cosh': cmath.cosh,
    'tanh': cmath.tanh,
    'asin': cmath.asin,
    'acos': cmath.acos,
    'atan': cmath.atan,
    'pi': cmath.pi,
    'e': cmath.e,
    'i': 1j,
    'j': 1j,
}

_MATH_FUNCS = [
    'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'conj',
    'sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan'
]

_ACCELERATED_FUNCS = {'sin', 'cos', 'exp', 'conj', 'abs'}
_ALLOWED_NAMES = {'z', 'c'}
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.UAdd,
    ast.USub,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
)


def normalize_formula(formula_str):
    """Convert user math notation into valid Python expression syntax."""
    normalized = formula_str.strip().replace('^', '**')
    normalized = re.sub(r'(\d)([zc])', r'\1*\2', normalized)
    normalized = re.sub(r'([zc])([zc])', r'\1*\2', normalized)
    normalized = re.sub(r'\)([zc(])', r')*\1', normalized)

    for func in _MATH_FUNCS:
        normalized = normalized.replace(func + '(', f'__FUNC_{func}__(')

    normalized = re.sub(r'([zc])(\()', r'\1*\2', normalized)

    for func in _MATH_FUNCS:
        normalized = normalized.replace(f'__FUNC_{func}__(', func + '(')

    return normalized


def _validate_accelerated_ast(normalized_formula):
    """Validate formula AST against accelerated-mode supported syntax."""
    try:
        expression = ast.parse(normalized_formula, mode='eval')
    except SyntaxError as exc:
        raise CustomFormulaError(f'Syntax error: {exc.msg}') from exc

    for node in ast.walk(expression):
        if not isinstance(node, _ALLOWED_NODES):
            raise CustomFormulaError('Unsupported syntax for accelerated formulas')

        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES | _ACCELERATED_FUNCS:
            raise CustomFormulaError(f'Unknown symbol: {node.id}')

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise CustomFormulaError('Only direct function calls are supported')
            if node.func.id not in _ACCELERATED_FUNCS:
                raise CustomFormulaError(
                    f'Unsupported function: {node.func.id}. '
                    'Allowed: sin, cos, exp, conj, abs'
                )
            if len(node.args) != 1:
                raise CustomFormulaError('Functions must take a single argument')


def _build_numba_expression(normalized_formula):
    """Map normalized formula names to NumPy/Numba-compatible symbols."""
    mapped = normalized_formula
    mapped = re.sub(r'\bconj\s*\(', 'np.conj(', mapped)
    mapped = re.sub(r'\babs\s*\(', 'np.abs(', mapped)
    mapped = re.sub(r'\bsin\s*\(', 'np.sin(', mapped)
    mapped = re.sub(r'\bcos\s*\(', 'np.cos(', mapped)
    mapped = re.sub(r'\bexp\s*\(', 'np.exp(', mapped)
    return mapped


def _build_numba_iter_function(normalized_formula):
    """Generate and JIT-compile a custom iteration function for z and c."""
    numba_expr = _build_numba_expression(normalized_formula)
    source = (
        'def _generated_custom_iter(z, c):\n'
        f'    return {numba_expr}\n'
    )

    namespace = {'np': np}
    exec(source, namespace)
    py_func = namespace['_generated_custom_iter']
    jit_func = numba.njit(fastmath=True)(py_func)

    # Trigger compilation early so unsupported operations fail here.
    jit_func(0j, 0j)
    return jit_func


def prepare_custom_formula(formula_str):
    """Normalize and compile a custom formula string."""
    normalized = normalize_formula(formula_str)
    if not normalized:
        raise CustomFormulaError('Formula is empty')

    try:
        compiled_code = compile(normalized, '<custom_formula>', 'eval')
    except SyntaxError as exc:
        raise CustomFormulaError(f'Syntax error: {exc.msg}') from exc

    _validate_accelerated_ast(normalized)

    try:
        jit_iter_func = _build_numba_iter_function(normalized)
    except Exception:
        jit_iter_func = None

    return PreparedCustomFormula(
        original=formula_str,
        normalized=normalized,
        compiled_code=compiled_code,
        jit_iter_func=jit_iter_func,
    )


def eval_prepared_formula(prepared_formula, z, c, local_scope=None):
    """Evaluate a pre-compiled formula at the given z and c values."""
    if local_scope is None:
        local_scope = {'z': z, 'c': c}
    else:
        local_scope['z'] = z
        local_scope['c'] = c

    try:
        return eval(prepared_formula.compiled_code, SAFE_EVAL_GLOBALS, local_scope)
    except Exception:
        return complex(1e10, 0)


def validate_custom_formula(formula_str):
    """
    Validate user custom formula before rendering.

    Returns:
        Tuple of (prepared_formula_or_none, error_message_or_none)
    """
    try:
        prepared = prepare_custom_formula(formula_str)
    except CustomFormulaError as exc:
        return None, str(exc)

    test_scope = {'z': 0j, 'c': 0j}
    test_points = [
        (0j, 0j),
        (0.2 + 0.3j, -0.7 + 0.27015j),
    ]

    for z, c in test_points:
        test_scope['z'] = z
        test_scope['c'] = c
        try:
            value = eval(prepared.compiled_code, SAFE_EVAL_GLOBALS, test_scope)
        except NameError as exc:
            return None, f'Unknown symbol: {exc}'
        except Exception as exc:
            return None, f'Invalid formula: {exc}'

        try:
            _ = complex(value)
        except Exception:
            return None, 'Formula must return a numeric value'

    return prepared, None
