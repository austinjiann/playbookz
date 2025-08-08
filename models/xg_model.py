import json, math
from pathlib import Path
import numpy as np

GOAL_X, GOAL_Y = 120.0, 40.0
POST_Y_LOW, POST_Y_HIGH = 36.0, 44.0

_COEFFS = None

def _load_coeffs():
    global _COEFFS
    if _COEFFS is None:
        coeffs_path = Path(__file__).resolve().parent.parent / 'data' / 'xg_coeffs.json'
        with open(coeffs_path, 'r') as f:
            _COEFFS = json.load(f)
    return _COEFFS

def _features(x, y, body_part: str, is_set_piece: bool, is_open_play: bool):
    x = float(x); y = float(y)
    dx = (GOAL_X - x)
    dist = (dx**2 + (GOAL_Y - y)**2) ** 0.5
    a1 = math.atan2(y - POST_Y_LOW, dx)
    a2 = math.atan2(y - POST_Y_HIGH, dx)
    angle = abs(a1 - a2)

    bp = (body_part or 'other').lower()
    return {
        'dist': dist,
        'angle': angle,
        'bp_left': 1 if 'left' in bp else 0,
        'bp_right': 1 if 'right' in bp else 0,
        'bp_head': 1 if 'head' in bp else 0,
        'open_play': 1 if is_open_play else 0,
        'set_piece': 1 if is_set_piece else 0,
    }

def predict_xg(*, x, y, body_part='other', is_set_piece=False, is_open_play=True) -> float:
    coeffs = _load_coeffs()
    feats = _features(x, y, body_part, is_set_piece, is_open_play)
    X = np.array([feats[k] for k in coeffs['columns']], dtype=float)
    z = coeffs['intercept'] + float(np.dot(np.array(coeffs['coef']), X))
    return 1.0 / (1.0 + math.exp(-z))