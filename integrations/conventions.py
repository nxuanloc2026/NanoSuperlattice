"""Units, frames and vector conventions shared by every exchange file.

A holds the direct lattice vectors in rows (nm). Then
  fractional row f      -> Cartesian x = f @ A
  direction [uvw]       -> A.T @ [u, v, w]
  plane normal (hkl)    -> inv(A) @ [h, k, l]
  reciprocal vector G   -> 2*pi * inv(A) @ [h, k, l]       (rad/nm)
Particle rotations R are active (x_SL = R @ x_NC, G_SL = R @ G_NC). A
separate proper rotation U maps SL Cartesian coordinates into the laboratory
frame (x_lab = U @ x_SL). U is never folded into a particle rotation.
"""

import numpy as np

from crystal_geometry import index_vector

HC_KEV_NM = 1.2398419843320026  # h*c in keV*nm (CODATA 2018)

# Multiplicative factors that convert a q column to rad/nm.
Q_UNITS = {
    "nm^-1": 1.0, "1/nm": 1.0, "nm-1": 1.0, "rad/nm": 1.0,
    "a^-1": 10.0, "1/a": 10.0, "a-1": 10.0, "rad/a": 10.0,
    "angstrom^-1": 10.0, "1/angstrom": 10.0,
    # Tools that report 1/d (cycles per length) omit the 2*pi factor.
    "nm^-1 cycles": 2 * np.pi, "cycles/nm": 2 * np.pi,
    "a^-1 cycles": 20 * np.pi, "cycles/a": 20 * np.pi,
}
ANGLE_UNITS = {"2theta_deg": np.pi / 180, "2theta_rad": 1.0}


def _normalize_unit(unit):
    return " ".join(str(unit).strip().lower().replace("å", "a").replace("⁻¹", "^-1").split())


def to_q_nm_inv(values, unit, wavelength_nm=None):
    """Convert q, 1/d or a measured 2-theta column to q in rad/nm.

    For angles, q = 4*pi*sin(theta)/lambda where the measured angle is 2*theta.
    """
    values = np.asarray(values, dtype=float)
    key = _normalize_unit(unit)
    if key in Q_UNITS:
        return values * Q_UNITS[key]
    if key in ANGLE_UNITS:
        if wavelength_nm is None or not wavelength_nm > 0:
            raise ValueError("Converting 2-theta to q requires a positive wavelength_nm.")
        theta = 0.5 * values * ANGLE_UNITS[key]
        return 4 * np.pi * np.sin(theta) / wavelength_nm
    known = ", ".join(sorted(set(Q_UNITS) | set(ANGLE_UNITS)))
    raise ValueError(f"Unknown q unit {unit!r}. Use one of: {known}.")


def two_theta_deg(q_nm_inv, wavelength_nm):
    """Elastic scattering angle 2*theta in degrees, NaN beyond the Ewald limit."""
    s = np.asarray(q_nm_inv, dtype=float) * wavelength_nm / (4 * np.pi)
    with np.errstate(invalid="ignore"):
        return np.where(np.abs(s) <= 1, 2 * np.degrees(np.arcsin(np.clip(s, -1, 1))), np.nan)


def energy_kev(wavelength_nm):
    return HC_KEV_NM / wavelength_nm


def direction(cell, uvw):
    return np.asarray(cell, float).T @ np.asarray(uvw, float)


def plane_normal(cell, hkl):
    return np.linalg.solve(np.asarray(cell, float), np.asarray(hkl, float))


def reciprocal_vectors(cell, hkl):
    """G = 2*pi*inv(A) @ h for each row of hkl, returned as rows (rad/nm)."""
    hkl = np.atleast_2d(np.asarray(hkl, float))
    return 2 * np.pi * hkl @ np.linalg.inv(np.asarray(cell, float)).T


def check_rotation(matrix, label="rotation", atol=2e-6):
    """Return R as an array after checking R.T @ R = I and det(R) = +1."""
    r = np.asarray(matrix, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError(f"{label} must be a finite 3x3 matrix.")
    if not np.allclose(r.T @ r, np.eye(3), atol=atol, rtol=0):
        raise ValueError(f"{label} is not orthogonal (R.T @ R != I).")
    if not np.isclose(np.linalg.det(r), 1.0, atol=atol, rtol=0):
        raise ValueError(f"{label} must be proper (det = +1).")
    return r


def laboratory_matrix(config, sl_cell):
    """Return U with x_lab = U @ x_SL from the optional top-level 'laboratory' block.

    Either {"matrix": U} or {"z": <SL axis>, "x": <SL axis>} where each axis is
    {"direction": [u,v,w]} or {"plane": [h,k,l]} in the superlattice basis.
    Lab z is taken exactly along the first axis; lab x is the component of the
    second axis perpendicular to z. For BornAgain GISAXS choose z along the
    substrate normal and x along the in-plane beam direction. Default: U = I.
    """
    lab = config.get("laboratory")
    if lab is None:
        return np.eye(3)
    if not isinstance(lab, dict):
        raise ValueError("laboratory must be a JSON object.")
    if "matrix" in lab:
        return check_rotation(lab["matrix"], "laboratory.matrix")
    if not {"z", "x"} <= set(lab):
        raise ValueError("laboratory needs either 'matrix' or both 'z' and 'x' SL axes.")
    z = index_vector(lab["z"], sl_cell)
    x = index_vector(lab["x"], sl_cell)
    x = x - z * (x @ z)
    if np.linalg.norm(x) < 1e-7:
        raise ValueError("laboratory.x must not be parallel to laboratory.z.")
    x /= np.linalg.norm(x)
    return np.vstack((x, np.cross(z, x), z))
