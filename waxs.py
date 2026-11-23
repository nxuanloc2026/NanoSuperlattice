"""Kinematic X-ray Bragg scattering from oriented nanocrystals.

q is the scattering-vector magnitude in radians per nanometer (2π/d).
Particle variants add incoherently in proportion to the number of sites.
Peak widths are Cartesian Gaussian standard deviations in nm⁻¹.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def _tabulated_factors():
    """DABAX Waasmaier–Kirfel coefficients, retaining the published species labels."""
    table = {}
    source = Path(__file__).with_name("f0_WaasKirf.dat")
    for line in source.read_text(encoding="ascii").splitlines():
        if line.startswith("#S "):
            symbol = line.split()[2]
        elif line.strip() and not line.startswith("#"):
            values = np.array(line.split(), dtype=float)
            if len(values) != 11:
                raise ValueError(f"Invalid form-factor data for {symbol} in {source.name}.")
            table[symbol] = (values[:5], values[6:], values[5])
    return table


def _atomic_factor(symbol, q, overrides):
    if symbol in overrides:
        value = overrides[symbol]
        if not isinstance(value, dict):
            constant = float(value)
            if not np.isfinite(constant):
                raise ValueError(f"The form factor for {symbol} must be finite.")
            return np.full_like(q, constant)
        a = np.asarray(value["a"], dtype=float)
        b = np.asarray(value["b"], dtype=float)
        c = float(value["c"])
        if (a.ndim != 1 or not a.size or b.shape != a.shape
                or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(c)):
            raise ValueError(f"Invalid Gaussian form-factor coefficients for {symbol}.")
    else:
        table = _tabulated_factors()
        if symbol not in table:
            raise ValueError(f"No tabulated X-ray form factor for {symbol!r}. Supply a number "
                             "or Gaussian coefficients in scattering.waxs.form_factors.")
        if np.any(q > 240 * np.pi):
            raise ValueError("Tabulated Waasmaier–Kirfel factors are valid to sinθ/λ = 6 Å⁻¹. "
                             "Reduce the WAXS range or supply appropriate form factors.")
        a, b, c = table[symbol]
    # q is in nm⁻¹. The coefficients use s = sin(theta)/lambda in Å⁻¹.
    s_squared = (q / (40 * np.pi)) ** 2
    return c + np.exp(-s_squared[:, None] * b) @ a


def _basis_sites(atomic):
    """Use each periodic basis site once. Viewer boundary copies do not scatter twice."""
    sites = {}
    for atom in atomic["basis"]:
        pos = np.asarray(atom["frac"], dtype=float) % 1.0
        key = (atom["element"], tuple(np.round(pos, 10) % 1.0))
        occupancy = float(atom.get("occupancy", 1.0))
        b_iso = float(atom.get("b_iso_nm2", 0.0))
        if not (0 <= occupancy <= 1 and np.isfinite(b_iso) and b_iso >= 0):
            raise ValueError("Atomic occupancies must be in [0,1] and b_iso_nm2 finite and ≥0.")
        if key in sites and sites[key][1:] != (occupancy, b_iso):
            raise ValueError("Periodic duplicates of the same species have inconsistent weights.")
        sites[key] = (pos, occupancy, b_iso)
    return [(key[0], *site) for key, site in sites.items()]


def _reflections(atomic, cell, limit, overrides):
    # h_i = a_i · G / (2π), so these bounds remain valid in skew cells.
    bounds = np.floor(np.linalg.norm(cell, axis=1) * limit / (2 * np.pi) + 1e-10).astype(int)
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    k, l = np.meshgrid(np.arange(-bounds[1], bounds[1]+1),
                       np.arange(-bounds[2], bounds[2]+1), indexing="ij")
    slab = np.column_stack((np.zeros(k.size, dtype=int), k.ravel(), l.ravel()))
    sites = _basis_sites(atomic)
    symbols = {site[0] for site in sites}
    # Validate species even when no reciprocal points fall in the requested window.
    for symbol in symbols:
        _atomic_factor(symbol, np.array([0.0]), overrides)
    indices, vectors, intensities = [], [], []
    for h in range(-bounds[0], bounds[0]+1):
        slab[:, 0] = h
        g = slab @ reciprocal
        length = np.linalg.norm(g, axis=1)
        mask = (length > 0) & (length <= limit + 1e-10)
        if not np.any(mask):
            continue
        hkl, g, q = slab[mask], g[mask], length[mask]
        factors = {s: _atomic_factor(s, q, overrides) for s in symbols}
        amplitude = np.zeros(len(q), dtype=complex)
        upper = np.zeros(len(q))
        for symbol, position, occupancy, b_iso in sites:
            f = occupancy * factors[symbol] * np.exp(-b_iso * (q / (4 * np.pi)) ** 2)
            amplitude += f * np.exp(2j * np.pi * (hkl @ position))
            upper += np.abs(f)
        # Remove only cancellations at numerical precision, on a per-reflection scale.
        allowed = np.abs(amplitude) > 1e-12 * upper
        if np.any(allowed):
            indices.append(hkl[allowed])
            vectors.append(g[allowed])
            intensities.append(np.abs(amplitude[allowed]) ** 2)
    if not indices:
        return np.empty((0, 3), int), np.empty((0, 3)), np.empty(0)
    return np.concatenate(indices), np.concatenate(vectors), np.concatenate(intensities)


def _shells(hkl, vectors, weights):
    """Merge equal d-spacings, including accidental degeneracies in a general cell."""
    q = np.linalg.norm(vectors, axis=1)
    peaks = []
    for i in np.argsort(q):
        indices = hkl[i].tolist()
        if not peaks or abs(q[i] - peaks[-1]["q"]) > max(1e-9, 1e-10 * q[i]):
            peaks.append({"hkl": indices, "q": float(q[i]), "d": float(2*np.pi/q[i]),
                          "intensity": float(weights[i]), "multiplicity": 1})
        else:
            peak = peaks[-1]
            peak["intensity"] += float(weights[i])
            peak["multiplicity"] += 1
            peak["hkl"] = max(peak["hkl"], indices)
    return peaks


def _powder_profile(q, peaks, sigma):
    """Exact orientational mean of each unit-integral isotropic 3D Gaussian."""
    result = np.zeros_like(q)
    for p in peaks:
        length = p["q"]
        x = 2*q*length / sigma**2
        correction = np.ones_like(q)
        np.divide(-np.expm1(-x), x, out=correction, where=x != 0)
        result += p["intensity"] * np.exp(-0.5*((q-length)/sigma)**2) * correction
    return result / (np.sqrt(2*np.pi) * sigma)**3


def simulate_waxs(scene, config, q_axis, ex, ey, radial_q, settings):
    """Return a central reciprocal-plane map, powder mean I(q), and powder peak list.

    ex, ey are orthonormal SL Cartesian unit vectors. Map rows increase along
    +ey. Map axes and powder q are in nm⁻¹; returned intensities use the same
    arbitrary scale. Integrated Bragg weights sum |F|² over the shell. The
    powder mean is defined per reciprocal-space volume; integrating it with
    4πq² dq recovers the shell weight. The forward (000) beam is omitted.

    settings accepts sigma_nm_inv and form_factors. Atomic sites optionally
    specify occupancy (default 1) and isotropic b_iso_nm2 (default 0).
    """
    sigma = float(settings.get("sigma_nm_inv", 0.45))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("WAXS sigma_nm_inv must be finite and positive.")
    axis = np.asarray(q_axis, dtype=float)
    radial = np.asarray(radial_q, dtype=float)
    ex, ey = np.asarray(ex, dtype=float), np.asarray(ey, dtype=float)
    if (axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all()
            or not np.all(np.diff(axis) > 0) or radial.ndim != 1 or not radial.size
            or not np.isfinite(radial).all() or np.any(radial < 0)):
        raise ValueError("WAXS needs a finite increasing map axis and nonnegative powder q.")
    if (ex.shape != (3,) or ey.shape != (3,)
            or not np.allclose([ex @ ex, ey @ ey, ex @ ey], [1, 1, 0], atol=1e-8, rtol=0)):
        raise ValueError("WAXS reciprocal-plane axes must be orthonormal.")
    overrides = settings.get("form_factors", {})
    if not isinstance(overrides, dict):
        raise ValueError("WAXS form_factors must map species labels to numbers or coefficients.")
    window = 6 * sigma
    limit = max(np.sqrt(2) * np.max(np.abs(axis)), radial.max()) + window
    hkl, vectors, weights = _reflections(config["atomic"], np.asarray(scene["ncCell"]),
                                         limit, overrides)
    peaks = _shells(hkl, vectors, weights)
    profile = _powder_profile(radial, peaks, sigma)
    image = np.zeros((len(axis), len(axis)))
    counts = np.bincount([s["variant"] for s in scene["centers"]], minlength=len(scene["variants"]))
    frame = np.column_stack((ex, ey, np.cross(ex, ey)))
    for variant, fraction in zip(scene["variants"], counts/counts.sum()):
        if fraction == 0:
            continue
        projected = vectors @ np.asarray(variant["rotation"]).T @ frame
        nearby = ((np.abs(projected[:, 2]) <= window)
                  & (projected[:, :2].min(axis=1) >= axis[0] - window)
                  & (projected[:, :2].max(axis=1) <= axis[-1] + window))
        for (gx, gy, gz), weight in zip(projected[nearby], weights[nearby]):
            i0, i1 = axis.searchsorted(gx-window), axis.searchsorted(gx+window, side="right")
            j0, j1 = axis.searchsorted(gy-window), axis.searchsorted(gy+window, side="right")
            px = np.exp(-0.5*((axis[i0:i1]-gx)/sigma)**2)
            py = np.exp(-0.5*((axis[j0:j1]-gy)/sigma)**2)
            image[j0:j1, i0:i1] += fraction * weight * np.exp(-0.5*(gz/sigma)**2) * np.outer(py, px)
    image /= (np.sqrt(2*np.pi) * sigma)**3
    return image, profile, [p for p in peaks if p["q"] <= radial.max()+1e-10]
