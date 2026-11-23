"""Dans_Diffraction adapter: an independent check of atomic (WAXS) reflections.

Dans_Diffraction is optional (pip install Dans_Diffraction). The adapter only
calls three long-standing entry points -- Crystal(cif), Scatter.setup_scatter
and Scatter.intensity -- and generates the hkl list itself, so the comparison
does not depend on Dans' Cartesian frame. Pin the version you validate with:
``run_comparison`` records it. Untested here against a live installation;
treat the first run as the validation of the adapter itself.

Comparison order follows the integration guide: hkl and q first, then
reflection absences, then unbroadened |F|^2 ratios. Intensities are compared
after normalizing each set to its strongest reflection, because the two codes
use different absolute scales.
"""

import inspect
from pathlib import Path

import numpy as np

from crystal_geometry import lattice_matrix
from visualize_superlattice import build_scene
from waxs import _atomic_factor, _basis_sites

from .conventions import energy_kev, laboratory_matrix, reciprocal_vectors


def hkl_within(cell, q_max):
    """All nonzero integer hkl with |G| <= q_max (rad/nm), sorted by q."""
    bounds = np.floor(np.linalg.norm(cell, axis=1) * q_max / (2 * np.pi) + 1e-10).astype(int)
    grid = np.stack(np.meshgrid(*[np.arange(-b, b + 1) for b in bounds], indexing="ij"), -1)
    hkl = grid.reshape(-1, 3)
    q = np.linalg.norm(reciprocal_vectors(cell, hkl), axis=1)
    keep = (q > 0) & (q <= q_max + 1e-10)
    order = np.argsort(q[keep], kind="stable")
    return hkl[keep][order], q[keep][order]


def structure_factor_squared(atomic, hkl, overrides=None):
    """Kinematic |F(hkl)|^2 with the same factors and B_iso model as waxs.py."""
    cell = lattice_matrix(atomic["cell"])
    hkl = np.atleast_2d(hkl)
    q = np.linalg.norm(reciprocal_vectors(cell, hkl), axis=1)
    amplitude = np.zeros(len(hkl), complex)
    for symbol, position, occupancy, b_iso in _basis_sites(atomic):
        f = occupancy * _atomic_factor(symbol, q, overrides or {}) * np.exp(-b_iso * (q / (4*np.pi))**2)
        amplitude += f * np.exp(2j * np.pi * (hkl @ position))
    return np.abs(amplitude) ** 2


def dans_intensities(cif_path, hkl, wavelength_nm, *, setup_kwargs=None, module=None):
    """Return (|F|^2 from Dans, q from Dans in rad/nm, version string)."""
    if module is None:
        try:
            import Dans_Diffraction as module
        except ImportError as exc:
            raise ImportError("Install Dans_Diffraction to run this adapter: "
                              "python -m pip install Dans_Diffraction") from exc
    xtl = module.Crystal(str(cif_path))
    kwargs = {"scattering_type": "xray", "energy_kev": energy_kev(wavelength_nm)}
    try:  # Dans >= 3: use the same Waasmaier-Kirfel f0 as waxs.py, and stay quiet
        accepted = inspect.signature(xtl.Scatter.setup_scatter).parameters
    except (TypeError, ValueError):
        accepted = {}
    if "use_waaskirf" in accepted:
        kwargs["use_waaskirf"] = True
    if "output" in accepted:
        kwargs["output"] = False
    kwargs.update(setup_kwargs or {})
    xtl.Scatter.setup_scatter(**kwargs)
    intensity = np.asarray(xtl.Scatter.intensity(np.asarray(hkl)), dtype=float)
    q = None
    if hasattr(xtl.Cell, "Qmag"):
        q = 10 * np.asarray(xtl.Cell.Qmag(np.asarray(hkl)), dtype=float)  # 1/A -> 1/nm
    return intensity, q, getattr(module, "__version__", "unknown")


def compare_reflections(hkl, q_ref, i_ref, i_ext, q_ext=None, *, absent=1e-8, rtol=0.02):
    """Compare two |F|^2 lists for the same hkl. Returns a report dictionary."""
    i_ref, i_ext = np.asarray(i_ref, float), np.asarray(i_ext, float)
    r = i_ref / i_ref.max() if i_ref.max() > 0 else i_ref
    e = i_ext / i_ext.max() if i_ext.max() > 0 else i_ext
    absent_ref, absent_ext = r < absent, e < absent
    present = ~absent_ref & ~absent_ext
    ratio = np.divide(e, r, out=np.full_like(r, np.nan), where=present)
    worst = int(np.nanargmax(np.abs(ratio - 1))) if present.any() else None
    report = {
        "reflections": int(len(hkl)),
        "absences_agree": bool(np.array_equal(absent_ref, absent_ext)),
        "absent_only_in_reference": np.asarray(hkl)[absent_ref & ~absent_ext].tolist(),
        "absent_only_in_external": np.asarray(hkl)[absent_ext & ~absent_ref].tolist(),
        "max_relative_intensity_error": float(np.nanmax(np.abs(ratio - 1))) if present.any() else 0.0,
        "worst_hkl": np.asarray(hkl)[worst].tolist() if worst is not None else None,
    }
    if q_ext is not None:
        report["max_q_error_nm_inv"] = float(np.max(np.abs(np.asarray(q_ext) - q_ref)))
    report["passed"] = bool(report["absences_agree"] and report["max_relative_intensity_error"] <= rtol
                            and report.get("max_q_error_nm_inv", 0.0) <= 1e-4 * max(q_ref.max(), 1))
    return report


def oriented_spots(config, hkl, intensity, scene=None):
    """Reflection vectors in the laboratory frame for each variant.

    Returns a list of (variant name, G_lab rows, intensity * population fraction),
    reproducing the incoherent variant weighting of waxs.py.
    """
    scene = scene or build_scene(config)
    nc = np.asarray(scene["ncCell"])
    lab = laboratory_matrix(config, np.asarray(scene["slCell"]))
    counts = np.bincount([c["variant"] for c in scene["centers"]], minlength=len(scene["variants"]))
    g_nc = reciprocal_vectors(nc, hkl)
    return [(v["name"], g_nc @ np.asarray(v["rotation"]).T @ lab.T,
             np.asarray(intensity) * counts[i] / counts.sum())
            for i, v in enumerate(scene["variants"])]


def powder_lines(hkl, q, intensity, tol=1e-9):
    """Merge equal spacings; returns rows (q, multiplicity, summed weight, representative hkl)."""
    lines = []
    for i in np.argsort(q, kind="stable"):
        if intensity[i] <= 0:
            continue
        if lines and abs(q[i] - lines[-1][0]) <= max(tol, 1e-10 * q[i]):
            lines[-1][1] += 1
            lines[-1][2] += float(intensity[i])
            lines[-1][3] = max(lines[-1][3], list(hkl[i]))
        else:
            lines.append([float(q[i]), 1, float(intensity[i]), list(hkl[i])])
    return lines


def run_comparison(config, cif_path, q_max_nm_inv=40.0, *, setup_kwargs=None, module=None):
    """Full Dans_Diffraction check for one configuration."""
    atomic = config["atomic"]
    wavelength = float(config.get("scattering", {}).get("wavelength_nm", 0.15406))
    cell = lattice_matrix(atomic["cell"])
    hkl, q = hkl_within(cell, q_max_nm_inv)
    overrides = config.get("scattering", {}).get("waxs", {}).get("form_factors", {})
    reference = structure_factor_squared(atomic, hkl, overrides)
    external, q_ext, version = dans_intensities(Path(cif_path), hkl, wavelength,
                                                setup_kwargs=setup_kwargs, module=module)
    report = compare_reflections(hkl, q, reference, external, q_ext)
    report.update(backend="Dans_Diffraction", backend_version=version,
                  wavelength_nm=wavelength, q_max_nm_inv=q_max_nm_inv)
    report["table"] = [{"hkl": h.tolist(), "q_nm_inv": float(qq), "d_nm": float(2*np.pi/qq),
                        "F2_reference": float(a), "F2_external": float(b)}
                       for h, qq, a, b in zip(hkl, q, reference, external)]
    return report
