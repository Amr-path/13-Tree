#!/usr/bin/env python
# coding: utf-8
"""
Tree Geometry and SAT Collision Detection Module
Extracted from santa_robust_solver.py for use in manual tree editor.

This module provides:
- Tree polygon loading and centering
- Placement class for storing tree positions/rotations
- Rotation caching for performance
- SAT (Separating Axis Theorem) collision detection
- AABB (Axis-Aligned Bounding Box) prefiltering
- Utility functions for computing bounds, sides, etc.
"""

import numpy as np
import pandas as pd
import os
import json
from typing import List, Tuple, Optional, Dict, Any

# Default tree polygon (will be overwritten if tree.csv exists)
TREE_DEFAULT = np.array([
    [0.0, 1.0], [-0.25, 0.6], [-0.15, 0.6], [-0.35, 0.3], [-0.20, 0.3],
    [-0.45, 0.0], [-0.10, 0.0], [-0.10, -0.20], [0.10, -0.20], [0.10, 0.0],
    [0.45, 0.0], [0.20, 0.3], [0.35, 0.3], [0.15, 0.6], [0.25, 0.6],
], dtype=np.float64)

TREE = TREE_DEFAULT.copy()

# Try to load from Kaggle or local paths
for path in ['/kaggle/input/santa-2025',
             '/kaggle/input/santa-2025-christmas-tree-packing-challenge',
             '.', '..']:
    tree_file = os.path.join(path, 'tree.csv')
    if os.path.exists(tree_file):
        try:
            df = pd.read_csv(tree_file)
            if 'x' in df.columns and 'y' in df.columns:
                TREE = df[['x', 'y']].values.astype(np.float64)
                print(f"Loaded tree from {tree_file}")
                break
        except:
            pass

# Center tree at origin
TREE = TREE - TREE.mean(axis=0)
TW = TREE[:, 0].max() - TREE[:, 0].min()  # Tree width
TH = TREE[:, 1].max() - TREE[:, 1].min()  # Tree height

def polygon_area(vertices: np.ndarray) -> float:
    """Calculate polygon area using shoelace formula."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0

TREE_AREA = polygon_area(TREE)


class Placement:
    """Represents a tree placement with position (x, y) and rotation (deg)."""
    __slots__ = ['x', 'y', 'deg']

    def __init__(self, x: float, y: float, deg: float):
        self.x = float(x)
        self.y = float(y)
        self.deg = float(deg) % 360

    def copy(self) -> 'Placement':
        return Placement(self.x, self.y, self.deg)

    def to_dict(self) -> Dict[str, float]:
        return {'x': self.x, 'y': self.y, 'deg': self.deg}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'Placement':
        return cls(d['x'], d['y'], d['deg'])

    def __repr__(self) -> str:
        return f"Placement(x={self.x:.4f}, y={self.y:.4f}, deg={self.deg:.1f})"


# Rotation cache for performance
ROTATION_CACHE: Dict[float, np.ndarray] = {}

def rotate_tree(deg: float) -> np.ndarray:
    """Get rotated tree vertices (cached)."""
    deg = float(deg) % 360
    deg_key = round(deg, 1)
    if deg_key not in ROTATION_CACHE:
        rad = np.radians(deg)
        c, s = np.cos(rad), np.sin(rad)
        rot_mat = np.array([[c, -s], [s, c]])
        ROTATION_CACHE[deg_key] = (TREE @ rot_mat.T).copy()
    return ROTATION_CACHE[deg_key].copy()

def get_tree_at(p: Placement) -> np.ndarray:
    """Get tree vertices at given placement position and rotation."""
    return rotate_tree(p.deg) + np.array([p.x, p.y])

# Pre-cache common angles
for angle in list(range(0, 360, 5)) + [0, 180]:
    rotate_tree(angle)


# ==============================================================================
# SAT Collision Detection
# ==============================================================================

def get_normals(poly: np.ndarray) -> np.ndarray:
    """Get normalized edge normals for SAT collision check."""
    edges = np.roll(poly, -1, axis=0) - poly
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1
    return normals / lengths

def sat_collision(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """Check if two polygons collide using SAT (Separating Axis Theorem)."""
    all_normals = np.vstack([get_normals(poly1), get_normals(poly2)])
    for normal in all_normals:
        proj1 = poly1 @ normal
        proj2 = poly2 @ normal
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return False
    return True

def get_aabb(p: Placement) -> Tuple[np.ndarray, np.ndarray]:
    """Get AABB (Axis-Aligned Bounding Box) for a tree placement."""
    tree = get_tree_at(p)
    return tree.min(axis=0), tree.max(axis=0)

def aabb_overlap(min1: np.ndarray, max1: np.ndarray,
                  min2: np.ndarray, max2: np.ndarray) -> bool:
    """Check if two AABBs overlap."""
    if max1[0] < min2[0] or max2[0] < min1[0]:
        return False
    if max1[1] < min2[1] or max2[1] < min1[1]:
        return False
    return True

def check_collision(p1: Placement, p2: Placement) -> bool:
    """Check if two tree placements collide (AABB prefilter + SAT)."""
    t1 = get_tree_at(p1)
    t2 = get_tree_at(p2)

    # Fast AABB rejection
    b1_min, b1_max = t1.min(axis=0), t1.max(axis=0)
    b2_min, b2_max = t2.min(axis=0), t2.max(axis=0)

    if b1_max[0] < b2_min[0] or b2_max[0] < b1_min[0]:
        return False
    if b1_max[1] < b2_min[1] or b2_max[1] < b1_min[1]:
        return False

    return sat_collision(t1, t2)

def has_any_collision(placements: List[Placement]) -> bool:
    """Check if any pair of placements collide."""
    n = len(placements)
    for i in range(n):
        for j in range(i + 1, n):
            if check_collision(placements[i], placements[j]):
                return True
    return False

def find_collisions(placements: List[Placement]) -> List[Tuple[int, int]]:
    """Find all colliding pairs of placements."""
    collisions = []
    n = len(placements)
    for i in range(n):
        for j in range(i + 1, n):
            if check_collision(placements[i], placements[j]):
                collisions.append((i, j))
    return collisions

def count_collisions_for(placements: List[Placement], idx: int) -> int:
    """Count collisions for a single tree."""
    count = 0
    for j in range(len(placements)):
        if idx != j and check_collision(placements[idx], placements[j]):
            count += 1
    return count

def check_single_collision(placements: List[Placement], idx: int) -> bool:
    """Check if a single tree collides with any other tree."""
    for j in range(len(placements)):
        if idx != j and check_collision(placements[idx], placements[j]):
            return True
    return False


# ==============================================================================
# Utility Functions
# ==============================================================================

def compute_side(placements: List[Placement]) -> float:
    """Compute the side length (max of width and height) for placements."""
    if not placements:
        return 0.0
    all_pts = np.vstack([get_tree_at(p) for p in placements])
    width = all_pts[:, 0].max() - all_pts[:, 0].min()
    height = all_pts[:, 1].max() - all_pts[:, 1].min()
    return max(width, height)

def compute_bounds(placements: List[Placement]) -> Tuple[float, float, float, float]:
    """Compute bounding box (xmin, ymin, xmax, ymax) for placements."""
    if not placements:
        return (0, 0, 0, 0)
    all_pts = np.vstack([get_tree_at(p) for p in placements])
    return (all_pts[:, 0].min(), all_pts[:, 1].min(),
            all_pts[:, 0].max(), all_pts[:, 1].max())

def center_placements(placements: List[Placement]) -> List[Placement]:
    """Center placements around origin."""
    if not placements:
        return placements
    all_pts = np.vstack([get_tree_at(p) for p in placements])
    cx = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2
    cy = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2
    return [Placement(p.x - cx, p.y - cy, p.deg) for p in placements]

def theoretical_min_side(n: int) -> float:
    """Calculate theoretical minimum side length for n trees."""
    return np.sqrt(n * TREE_AREA)


# ==============================================================================
# CSV I/O Functions
# ==============================================================================

def parse_s_value(s: str) -> float:
    """Parse a value with 's' prefix (e.g., 's1.234567')."""
    if isinstance(s, str) and s.startswith('s'):
        return float(s[1:])
    return float(s)

def format_s_value(v: float) -> str:
    """Format a value with 's' prefix (e.g., 's1.234567')."""
    return f"s{v:.6f}"

def load_submission_csv(filepath: str) -> Dict[int, List[Placement]]:
    """Load placements from submission CSV file."""
    df = pd.read_csv(filepath)
    state = {}

    for _, row in df.iterrows():
        id_str = row['id']
        n = int(id_str.split('_')[0])
        x = parse_s_value(row['x'])
        y = parse_s_value(row['y'])
        deg = parse_s_value(row['deg'])

        if n not in state:
            state[n] = []
        state[n].append(Placement(x, y, deg))

    return state

def save_submission_csv(state: Dict[int, List[Placement]], filepath: str) -> None:
    """Save placements to submission CSV file."""
    rows = []
    for n in sorted(state.keys()):
        for i, p in enumerate(state[n]):
            rows.append({
                'id': f'{n:03d}_{i}',
                'x': format_s_value(p.x),
                'y': format_s_value(p.y),
                'deg': format_s_value(p.deg),
            })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)

def load_state_json(filepath: str) -> Dict[int, List[Placement]]:
    """Load placements from JSON state file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    state = {}
    for n_str, placements_data in data.items():
        n = int(n_str)
        state[n] = [Placement.from_dict(p) for p in placements_data]

    return state

def save_state_json(state: Dict[int, List[Placement]], filepath: str) -> None:
    """Save placements to JSON state file."""
    data = {}
    for n, placements in state.items():
        data[str(n)] = [p.to_dict() for p in placements]

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


# ==============================================================================
# Score Calculation
# ==============================================================================

def compute_score(state: Dict[int, List[Placement]], max_n: int = 200) -> float:
    """Compute total score for all groups."""
    total = 0.0
    for n in range(1, max_n + 1):
        if n in state:
            side = compute_side(state[n])
            total += side**2 / n
    return total

def compute_group_score(placements: List[Placement], n: int) -> float:
    """Compute score contribution for a single group."""
    side = compute_side(placements)
    return side**2 / n


# ==============================================================================
# Module Info
# ==============================================================================

print(f"Tree Geometry Module Loaded")
print(f"  Tree: {len(TREE)} vertices, {TW:.4f} x {TH:.4f}, area={TREE_AREA:.4f}")
print(f"  Cached rotations: {len(ROTATION_CACHE)} angles")
