"""Separate symmetry operations from physical orientation populations.

Proper lattice rotations permute symmetry-equivalent directions and leave the
reciprocal-lattice spot set unchanged. Two variants whose relative rotation
Delta = R_i.T @ R_j is such an operation are the same atomic orientation as
far as Bragg spot positions go. The particle envelope (or a termination or
surface marking) can still distinguish them; ``classify_variants`` reports
each level separately, and ``disorientation_deg`` gives the smallest physical
misorientation once crystal symmetry is factored out.
"""

from itertools import product

import numpy as np

from crystal_geometry import lattice_matrix, particle_mesh
from visualize_superlattice import build_scene
from waxs import _basis_sites


def lattice_rotations(cell, tol=1e-6):
    """Proper Cartesian rotations R with R @ A.T = A.T @ N for integer N (|N_ij| <= 1)."""
    a = np.asarray(cell, float)
    metric = a @ a.T
    candidates = np.array(list(product((-1, 0, 1), repeat=9))).reshape(-1, 3, 3)
    candidates = candidates[np.isclose(np.linalg.det(candidates), 1)]
    # N acts on fractional column vectors: f' = N f. Metric preserved: N.T G N = G with G = A A.T.
    keep = np.all(np.isclose(np.einsum("nji,jk,nkl->nil", candidates, metric, candidates),
                             metric, atol=tol * np.abs(metric).max()), axis=(1, 2))
    a_t, a_t_inv = a.T, np.linalg.inv(a.T)
    return [a_t @ n @ a_t_inv for n in candidates[keep]], list(candidates[keep])


def crystal_rotations(atomic, tol=1e-4):
    """Lattice rotations that, with some translation, map the atomic basis onto itself."""
    cell = lattice_matrix(atomic["cell"])
    sites = _basis_sites(atomic)
    species = np.array([s[0] for s in sites])
    frac = np.array([s[1] for s in sites])
    rotations, integer = lattice_rotations(cell)
    kept = []
    for r, n in zip(rotations, integer):
        mapped = frac @ n.T
        ref = mapped[0]
        for target in frac[species == species[0]]:
            shift = target - ref
            moved = mapped + shift
            ok = True
            for sp, p in zip(species, moved):
                d = (frac[species == sp] - p + 0.5) % 1.0 - 0.5
                if not np.any(np.all(np.abs(d) < tol, axis=1)):
                    ok = False
                    break
            if ok:
                kept.append(r)
                break
    return kept


def envelope_rotations(shape, candidates, tol=1e-6):
    """Candidates that map the particle mesh vertex set (about its center) onto itself."""
    if shape["type"] == "sphere":
        return list(candidates)
    vertices, _ = particle_mesh(shape)
    v = np.asarray(vertices, float)
    v = v - v.mean(axis=0)
    scale = np.abs(v).max()
    kept = []
    for r in candidates:
        moved = v @ np.asarray(r).T
        d = np.linalg.norm(moved[:, None] - v[None], axis=-1).min(axis=1)
        if np.all(d < tol * scale * 10):
            kept.append(r)
    return kept


def _angle_axis(delta):
    axis = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])
    # atan2 keeps precision near 0 and 180 degrees, where arccos(trace) does not.
    angle = np.degrees(np.arctan2(np.linalg.norm(axis) / 2, (np.trace(delta) - 1) / 2))
    if np.linalg.norm(axis) < 1e-9:  # 0 or 180 degrees
        w, vecs = np.linalg.eigh((delta + delta.T) / 2)
        axis = vecs[:, np.argmax(w)]
    return float(angle), axis / np.linalg.norm(axis)


def _member(delta, group, tol=1e-6):
    return any(np.allclose(delta, g, atol=tol) for g in group)


def disorientation_deg(r1, r2, group):
    """Smallest rotation angle between two orientations modulo crystal symmetry."""
    delta = np.asarray(r1).T @ np.asarray(r2)
    return min(_angle_axis(delta @ g)[0] for g in group)


def classify_variants(config, scene=None):
    """Pairwise relations between variants: lattice, crystal and envelope level."""
    scene = scene or build_scene(config)
    nc = np.asarray(scene["ncCell"])
    lattice, _ = lattice_rotations(nc)
    crystal = crystal_rotations(config["atomic"])
    report = {"lattice_operations": len(lattice), "crystal_operations": len(crystal), "pairs": []}
    for i, j in ((i, j) for i in range(len(scene["variants"])) for j in range(i + 1, len(scene["variants"]))):
        ri, rj = (np.asarray(scene["variants"][k]["rotation"]) for k in (i, j))
        delta = ri.T @ rj  # in the NC frame
        angle, axis = _angle_axis(delta)
        shape_i = config["variants"][i].get("shape", config["shape"])
        envelope = envelope_rotations(shape_i, crystal)
        same_spots = _member(delta, lattice)
        same_crystal = _member(delta, crystal)
        same_particle = same_crystal and _member(delta, envelope)
        axis_frac = np.linalg.solve(nc.T, axis)  # direction indices [uvw] of the axis
        axis_frac /= np.abs(axis_frac).max()
        report["pairs"].append({
            "variants": [scene["variants"][i]["name"], scene["variants"][j]["name"]],
            "relative_rotation_deg": angle, "axis_nc_uvw": np.round(axis_frac, 4).tolist(),
            "same_spot_positions": bool(same_spots), "same_crystal_orientation": bool(same_crystal),
            "indistinguishable_particles": bool(same_particle),
            "disorientation_deg": disorientation_deg(ri, rj, crystal),
            "verdict": ("identical particles (symmetry operation)" if same_particle else
                        "same atomic orientation; envelope distinguishes" if same_crystal else
                        "same spot positions; basis intensities differ" if same_spots else
                        "physically distinct atomic orientation")})
    return report
