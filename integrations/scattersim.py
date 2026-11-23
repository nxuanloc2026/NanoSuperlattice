"""Orientationally averaged SAXS of the finite assembly, for ScatterSim-type comparisons.

ScatterSim and most lattice-model codes report a circularly averaged curve of
an isotropic powder of superlattice grains. The viewer's SAXS profile is an
azimuthal mean inside one reciprocal-space slice, which differs whenever the
assembly is anisotropic. ``saxs_powder_profile`` evaluates the same coherent
amplitude as saxs.py, averaged over the full sphere of directions.
"""

import numpy as np

from saxs import _assembly_intensity
from visualize_superlattice import build_scene


def fibonacci_hemisphere(n):
    """n nearly uniform unit vectors on the upper hemisphere (Friedel symmetry)."""
    i = np.arange(n) + 0.5
    z = 1 - i / n  # uniform in cos(theta) on [0, 1)
    phi = np.pi * (1 + 5**0.5) * i
    r = np.sqrt(1 - z**2)
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def saxs_powder_profile(config, q, *, n_directions=500, scene=None):
    """Mean of |A(q n)|^2 over n_directions unit vectors n, for each |q|."""
    scene = scene or build_scene(config)
    q = np.asarray(q, float)
    settings = config.get("scattering", {}).get("saxs", {})
    contrast = np.asarray(settings.get("contrast", 1), float)
    if contrast.ndim == 0:
        contrast = np.repeat(contrast, len(scene["variants"]))
    directions = fibonacci_hemisphere(n_directions)
    vectors = (q[:, None, None] * directions[None]).reshape(-1, 3)
    return _assembly_intensity(scene, config, vectors, contrast).reshape(len(q), -1).mean(axis=1)
