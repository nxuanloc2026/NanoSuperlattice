"""Finite atomistic assemblies for Debye-equation and real-space codes.

Termination: the fractional point ``origin_frac`` of the atomic lattice
(default the cell origin, [0,0,0]) is placed at the particle center. Every
lattice translation of every unique basis site is kept when it lies inside
the particle envelope, including atoms within ``surface_tol_nm`` of a face.
Each particle j is then placed at x = R_j @ x_local + t_j.

    from integrations.atomistic import estimate, build_assembly
    print(estimate(config))            # check counts and memory first
    asm = build_assembly(config)       # dict of numpy arrays
"""

import numpy as np

from crystal_geometry import lattice_matrix, particle_mesh
from visualize_superlattice import build_scene
from waxs import _basis_sites

DEFAULT_MAX_ATOMS = 2_000_000


def _mesh_planes(vertices, faces):
    """Face planes n.x = d oriented so the vertex centroid lies inside."""
    centroid = vertices.mean(axis=0)
    normals, offsets = [], []
    for face in faces:
        p = vertices[list(face)]
        n = np.sum(np.cross(p, np.roll(p, -1, axis=0)), axis=0)  # Newell normal
        n /= np.linalg.norm(n)
        d = n @ p.mean(axis=0)
        if n @ centroid > d:
            n, d = -n, -d
        normals.append(n)
        offsets.append(d)
    return np.array(normals), np.array(offsets)


def _is_convex(vertices, normals, offsets, tol=1e-9):
    return bool(np.all(vertices @ normals.T <= offsets + tol * max(1, np.abs(offsets).max())))


def _winding_inside(points, vertices, faces, chunk=4096):
    """Generalized winding number (sum of triangle solid angles) > 0.5."""
    tris = np.array([(f[0], f[j], f[j + 1]) for f in faces for j in range(1, len(f) - 1)])
    inside = np.empty(len(points), bool)
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk, None, :]
        a, b, c = (vertices[tris[:, k]][None] - p for k in range(3))
        la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
        det = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc
               + np.einsum("ijk,ijk->ij", b, c) * la + np.einsum("ijk,ijk->ij", c, a) * lb)
        inside[start:start + chunk] = np.abs(np.arctan2(det, den).sum(axis=1)) / (2 * np.pi) > 0.5
    return inside


def inside_shape(points, shape, tol=1e-9):
    """Boolean mask of local points (nm) inside the envelope (surface inclusive)."""
    points = np.asarray(points, float)
    if shape["type"] == "sphere":  # the drawn sphere mesh is only an approximation
        return np.linalg.norm(points, axis=1) <= float(shape["radius_nm"]) + tol
    vertices, faces = particle_mesh(shape)
    vertices = np.asarray(vertices, float)
    normals, offsets = _mesh_planes(vertices, faces)
    if _is_convex(vertices, normals, offsets):
        return np.all(points @ normals.T <= offsets + tol, axis=1)
    return _winding_inside(points, vertices, faces)


def _shape_volume(shape):
    if shape["type"] == "sphere":
        return 4 / 3 * np.pi * float(shape["radius_nm"]) ** 3
    vertices, faces = particle_mesh(shape)
    v = np.asarray(vertices, float)
    tris = np.array([(f[0], f[j], f[j + 1]) for f in faces for j in range(1, len(f) - 1)])
    return abs(np.einsum("ij,ij->i", v[tris[:, 0]], np.cross(v[tris[:, 1]], v[tris[:, 2]])).sum() / 6)


def particle_atoms(atomic, shape, origin_frac=(0, 0, 0), surface_tol_nm=1e-9):
    """Atoms of one crystallite in its local NC frame.

    Returns positions (n,3) nm, species list, occupancy (n,), b_iso_nm2 (n,).
    """
    cell = lattice_matrix(atomic["cell"])
    vertices, _ = particle_mesh(shape)
    reach = float(np.max(np.linalg.norm(vertices, axis=1)))
    if shape["type"] == "sphere":
        reach = float(shape["radius_nm"])
    # |f_i| <= reach * |column i of inv(A)| bounds every fractional coordinate.
    bounds = np.ceil(reach * np.linalg.norm(np.linalg.inv(cell), axis=0)).astype(int) + 1
    grid = np.stack(np.meshgrid(*[np.arange(-b, b + 1) for b in bounds], indexing="ij"), -1).reshape(-1, 3)
    positions, species, occupancy, b_iso = [], [], [], []
    for symbol, frac, occ, b in _basis_sites(atomic):
        xyz = (grid + frac - np.asarray(origin_frac, float)) @ cell
        keep = inside_shape(xyz, shape, surface_tol_nm)
        positions.append(xyz[keep])
        species += [symbol] * int(keep.sum())
        occupancy.append(np.full(keep.sum(), occ))
        b_iso.append(np.full(keep.sum(), b))
    return (np.concatenate(positions), species, np.concatenate(occupancy), np.concatenate(b_iso))


def estimate(config, scene=None):
    """Expected atom count, Debye pair count and memory before building anything."""
    scene = scene or build_scene(config)
    atomic = config["atomic"]
    volume_cell = abs(np.linalg.det(lattice_matrix(atomic["cell"])))
    per_cell = sum(occ for _, _, occ, _ in _basis_sites(atomic))
    counts = np.bincount([c["variant"] for c in scene["centers"]], minlength=len(scene["variants"]))
    atoms = 0.0
    for i, variant in enumerate(config["variants"]):
        shape = variant.get("shape", config["shape"])
        atoms += counts[i] * _shape_volume(shape) * per_cell / volume_cell
    n = int(round(atoms))
    return {"particles": int(counts.sum()), "atoms_estimate": n,
            "debye_pairs": n * (n - 1) // 2,
            "positions_memory_mb": 8 * 3 * n / 1e6,
            "note": "Pair count sets the Debye cost; histogram mode bins these pairs."}


def _random_rotation(sigma_rad, rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.normal(scale=sigma_rad)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def build_assembly(config, scene=None, *, origin_frac=(0, 0, 0), disorder=None,
                   max_atoms=DEFAULT_MAX_ATOMS, overlap_nm=None, surface_tol_nm=1e-9):
    """Replicate, clip, rotate and place every particle.

    disorder: {"position_sigma_nm": s, "rotation_sigma_deg": a, "seed": n}
    applies independent Gaussian center shifts and small random rotations.
    Returns dict(positions, species, occupancy, b_iso_nm2, particle, checks).
    """
    scene = scene or build_scene(config)
    est = estimate(config, scene)
    if est["atoms_estimate"] > max_atoms:
        raise ValueError(f"About {est['atoms_estimate']:,} atoms exceeds max_atoms={max_atoms:,}. "
                         "Reduce repeat or particle size, or raise max_atoms deliberately.")
    disorder = disorder or {}
    rng = np.random.default_rng(disorder.get("seed"))
    pos_sigma = float(disorder.get("position_sigma_nm", 0.0))
    rot_sigma = np.radians(float(disorder.get("rotation_sigma_deg", 0.0)))
    templates = []
    for variant in config["variants"]:
        shape = variant.get("shape", config["shape"])
        templates.append(particle_atoms(config["atomic"], shape, origin_frac, surface_tol_nm))
    blocks, species, occ, biso, owner, centers, radii = [], [], [], [], [], [], []
    for j, center in enumerate(scene["centers"]):
        local, sp, o, b = templates[center["variant"]]
        rotation = np.asarray(scene["variants"][center["variant"]]["rotation"])
        if rot_sigma:
            rotation = _random_rotation(rot_sigma, rng) @ rotation
        t = np.asarray(center["xyz"]) + (rng.normal(scale=pos_sigma, size=3) if pos_sigma else 0)
        blocks.append(local @ rotation.T + t)
        species += sp
        occ.append(o)
        biso.append(b)
        owner.append(np.full(len(local), j))
        centers.append(t)
        radii.append(np.linalg.norm(local, axis=1).max() if len(local) else 0.0)
    positions = np.concatenate(blocks)
    owner = np.concatenate(owner)
    checks = {"atoms": int(len(positions)), "atoms_estimate": est["atoms_estimate"],
              "atoms_per_particle": [int(len(t[0])) for t in templates],
              "duplicate_atoms": int(len(positions) - len(np.unique(np.round(positions, 6), axis=0))),
              "occupancy_sum": float(np.concatenate(occ).sum())}
    # Default overlap threshold: 90 % of the intra-crystal nearest-neighbor distance.
    sample = templates[0][0][:600]
    if overlap_nm is None and len(sample) > 1:
        d = np.linalg.norm(sample[:, None] - sample[None], axis=-1)
        overlap_nm = 0.9 * float(d[d > 1e-9].min())
    overlap_nm = overlap_nm or 0.1
    checks["overlap_threshold_nm"] = overlap_nm
    # Only particle pairs whose bounding spheres intersect need atom distances.
    centers, radii = np.array(centers), np.array(radii)
    closest, pairs = np.inf, 0
    for a in range(len(centers)):
        near = np.flatnonzero(np.linalg.norm(centers[a + 1:] - centers[a], axis=1)
                              < radii[a] + radii[a + 1:] + overlap_nm) + a + 1
        for b in near:
            pairs += 1
            pa, pb = blocks[a], blocks[b]
            for s in range(0, len(pa), 2048):
                d = np.linalg.norm(pa[s:s + 2048, None] - pb[None], axis=-1).min()
                closest = min(closest, float(d))
    checks["touching_particle_pairs"] = pairs
    checks["closest_interparticle_atoms_nm"] = None if not np.isfinite(closest) else closest
    checks["overlap"] = bool(np.isfinite(closest) and closest < overlap_nm)
    return {"positions": positions, "species": species, "occupancy": np.concatenate(occ),
            "b_iso_nm2": np.concatenate(biso), "particle": owner, "checks": checks}
