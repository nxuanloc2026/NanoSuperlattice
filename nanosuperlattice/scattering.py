"""Small- and wide-angle kinematic X-ray scattering in a common sample frame."""

import numpy as np

from .crystal_geometry import index_vector
from .saxs import simulate_saxs
from .waxs import simulate_waxs


def _label(spec):
    kind, indices = next(iter(spec.items()))
    indices = " ".join(f"{v:g}" for v in indices)
    return f"[{indices}] SL" if kind == "direction" else f"normal to ({indices}) SL"


def _slice_frame(cell, view):
    """Make a right-handed frame with x horizontal and x cross y along the normal."""
    normal_spec = view.get("normal", {"plane": [0, 0, 1]})
    normal = index_vector(normal_spec, cell)
    if "horizontal" in view:
        horizontal = view["horizontal"]
        x = index_vector(horizontal, cell)
        if abs(x @ normal) > 1e-7:
            raise ValueError("scattering.view.horizontal must be in the specified slice plane.")
        x_label = _label(horizontal)
    else:
        # Choose a well-conditioned axis to project, even in a skew cell.
        rows = cell / np.linalg.norm(cell, axis=1, keepdims=True)
        i = int(np.argmin(np.abs(rows @ normal)))
        x = rows[i] - normal * (rows[i] @ normal)
        x_label = f"projected {'abc'[i]} SL"
    x = x / np.linalg.norm(x)
    return x, np.cross(normal, x), normal, _label(normal_spec), x_label


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number.")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return value


def _q_grid(options, q_max, pixels, samples):
    q_max = _positive(options.get("q_max_nm_inv", q_max), "q_max_nm_inv")
    pixels = options.get("pixels", pixels)
    samples = options.get("profile_points", samples)
    if type(pixels) is not int or pixels < 3 or pixels % 2 == 0:
        raise ValueError("pixels must be an odd integer ≥ 3 so the map samples q = 0.")
    if type(samples) is not int or samples < 2:
        raise ValueError("profile_points must be an integer ≥ 2.")
    return np.linspace(-q_max, q_max, pixels), np.linspace(0, q_max, samples)


def _profile_maxima(q, intensity):
    """Interior peaks of a uniformly sampled profile, refined by a local parabola."""
    peaks = []
    for i in np.flatnonzero((intensity[1:-1] > intensity[:-2]) &
                            (intensity[1:-1] >= intensity[2:])) + 1:
        left, center, right = intensity[i-1:i+2]
        offset = 0.5 * (left - right) / (left - 2 * center + right)
        position = float(q[i] + offset * (q[i] - q[i-1]))
        height = float(center + 0.25 * (right - left) * offset)
        peaks.append({"q": position, "d": float(2*np.pi / position), "intensity": height})
    return peaks


def simulate_scattering(config, scene=None):
    """Return raw SAXS/WAXS intensities and reciprocal axes as JSON-compatible data.

    image[iy][ix] is evaluated at axis[ix]*xCartesian + axis[iy]*yCartesian,
    where the unit vectors are expressed in SL Cartesian coordinates.
    All q values use the 2*pi/d convention with lengths in nm.
    """
    if scene is None:
        from .visualize_superlattice import build_scene
        scene = build_scene(config)

    options = config.get("scattering", {})
    if not isinstance(options, dict):
        raise ValueError("scattering must be a JSON object.")
    view = options.get("view", {})
    sa = options.get("saxs", {})
    wa = options.get("waxs", {})
    if not all(isinstance(value, dict) for value in (view, sa, wa)):
        raise ValueError("view, saxs and waxs settings must be JSON objects.")
    wavelength = _positive(options.get("wavelength_nm", 0.15406), "wavelength_nm")
    cell = np.asarray(scene["slCell"])
    x, y, normal, label, x_label = _slice_frame(cell, view)
    sax, sr = _q_grid(sa, 4 * 2 * np.pi / min(np.linalg.norm(cell, axis=1)), 121, 201)
    wax, wr = _q_grid(wa, 55.0, 241, 551)
    si, sp = simulate_saxs(scene, config, sax, x, y, sr, sa)
    wi, wp, peaks = simulate_waxs(scene, config, wax, x, y, wr, wa)
    return {
        "viewLabel": f"Slice normal: {label}", "xLabel": x_label, "yLabel": "slice normal × horizontal",
        "xCartesian": x.tolist(), "yCartesian": y.tolist(), "normalCartesian": normal.tolist(),
        "wavelengthNm": wavelength,
        "saxs": {"axis": sax.tolist(), "image": si.tolist(), "q": sr.tolist(),
                 "intensity": sp.tolist(), "profileLabel": "Azimuthal mean in the slice",
                 "maxima": _profile_maxima(sr, sp),
                 "note": "Coherent scattering from the finite set of displayed particles, "
                         "including their shapes, contrasts and orientations. The central "
                         "maximum includes forward scattering."},
        "waxs": {"axis": wax.tolist(), "image": wi.tolist(), "q": wr.tolist(),
                 "intensity": wp.tolist(), "peaks": peaks, "profileLabel": "Powder Bragg profile",
                 "note": "The map combines oriented atomic Bragg spots using the displayed "
                         "variant populations. The powder profile includes reflections in all "
                         f"directions. Isotropic Gaussian σ = {float(wa.get('sigma_nm_inv', 0.45)):g} nm⁻¹."},
    }
