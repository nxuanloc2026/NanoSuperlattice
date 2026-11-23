"""Crystallographic geometry for the nanocrystal superlattice viewer.

Cell matrices have direct basis vectors in rows. Orientation matrices are
active rotations: a Cartesian column vector transforms as x_SL = R @ x_NC.
"""

from collections import Counter
from itertools import permutations, product

import numpy as np


def _array(value, shape, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain real numbers") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must have shape {shape} and finite entries")
    return result


def lattice_matrix(cell):
    """Return a right-handed Cartesian cell from [a,b,c,alpha,beta,gamma].

    Lengths retain the input units. Angles are in degrees: alpha is between
    b and c, beta between a and c, and gamma between a and b.
    """
    values = _array(cell, (6,), "cell [a,b,c,alpha,beta,gamma]")
    lengths, angles = values[:3], values[3:]
    if np.any(lengths <= 0) or np.any((angles <= 0) | (angles >= 180)):
        raise ValueError("Cell lengths must be positive and angles between 0 and 180 degrees")
    ca, cb, cg = np.cos(np.radians(angles))
    sg = np.sin(np.radians(angles[2]))
    cy = (ca - cb * cg) / sg
    cz_squared = 1 - cb * cb - cy * cy
    if sg < 1e-10 or cz_squared <= 1e-14:
        raise ValueError("Cell angles must define a nondegenerate three-dimensional cell")
    unit_rows = np.array([[1, 0, 0], [cg, sg, 0], [cb, cy, np.sqrt(cz_squared)]])
    return lengths[:, None] * unit_rows


def index_vector(axis, cell_matrix):
    """Return the unit Cartesian direction or plane normal of a signed triplet."""
    cell = _array(cell_matrix, (3, 3), "lattice matrix")
    if not isinstance(axis, dict) or set(axis) not in ({"direction"}, {"plane"}):
        raise ValueError("Each axis requires exactly one 'direction' or 'plane' triplet")
    kind, raw = next(iter(axis.items()))
    indices = _array(raw, (3,), kind)
    if not np.any(indices):
        raise ValueError(f"The {kind} triplet cannot be zero")
    # [uvw] uses the direct basis; a normal to (hkl) uses the dual basis.
    try:
        cart = cell.T @ indices if kind == "direction" else np.linalg.solve(cell, indices)
    except np.linalg.LinAlgError as exc:
        raise ValueError("A plane normal requires an invertible lattice matrix") from exc
    length = np.linalg.norm(cart)
    if not np.isfinite(length) or length == 0:
        raise ValueError(f"The {kind} triplet does not define a finite nonzero Cartesian vector")
    return cart / length


def _frame(first, second):
    normal = np.cross(first, second)
    sine = np.linalg.norm(normal)
    if sine < 1e-7:
        raise ValueError("An orientation requires two nonparallel, non-antiparallel axes")
    normal /= sine
    return np.column_stack((first, np.cross(normal, first), normal))


def orientation_matrix(spec, nc_cell_matrix, sl_cell_matrix):
    """Construct an active rotation from NC Cartesian to SL Cartesian axes.

    Supply either {"matrix": R} or {"align": [pair1, pair2]}. Each pair
    has "nc" and "sl" entries. Each entry is {"direction": [u,v,w]} or
    {"plane": [h,k,l]}. Signs specify directed alignment, including for
    plane normals. Both axis pairs must subtend the same Cartesian angle.
    """
    if not isinstance(spec, dict) or set(spec) not in ({"matrix"}, {"align"}):
        raise ValueError("Define an orientation with exactly one 'matrix' or 'align' key")
    if "matrix" in spec:
        rotation = _array(spec["matrix"], (3, 3), "orientation matrix")
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-6, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1, atol=2e-6, rtol=0)):
            raise ValueError("The orientation matrix must be orthogonal with determinant +1")
        return rotation

    pairs = spec["align"]
    if not isinstance(pairs, (list, tuple)) or len(pairs) != 2:
        raise ValueError("'align' needs two pairs of indexed axes")
    if any(not isinstance(pair, dict) or set(pair) != {"nc", "sl"} for pair in pairs):
        raise ValueError("Each alignment pair needs an 'nc' and an 'sl' axis")
    local_cell = _array(nc_cell_matrix, (3, 3), "NC lattice matrix")
    host_cell = _array(sl_cell_matrix, (3, 3), "SL lattice matrix")
    local = [index_vector(pair["nc"], local_cell) for pair in pairs]
    host = [index_vector(pair["sl"], host_cell) for pair in pairs]
    rotation = _frame(*host) @ _frame(*local).T
    if not np.allclose(rotation @ np.array(local).T, np.array(host).T, atol=1e-6, rtol=0):
        angles = [np.degrees(np.arccos(np.clip(v[0] @ v[1], -1, 1))) for v in (local, host)]
        raise ValueError(
            f"Alignment angles differ (NC {angles[0]:.6g}°, SL {angles[1]:.6g}°); "
            "a rigid rotation must preserve them"
        )
    return rotation


def _support_faces(vertices, normals):
    faces = []
    for raw in normals:
        normal = raw / np.linalg.norm(raw)
        scores = vertices @ normal
        indices = np.flatnonzero(np.isclose(scores, scores.max()))
        points = vertices[indices] - vertices[indices].mean(axis=0)
        first = points[0] / np.linalg.norm(points[0])
        second = np.cross(normal, first)
        order = np.argsort(np.arctan2(points @ second, points @ first))
        faces.append(tuple(int(i) for i in indices[order]))
    return faces


def _sphere():
    longitude_count, latitude_count = 12, 6
    vertices = [[0, 0, 1]]
    for latitude in range(1, latitude_count):
        theta = np.pi * latitude / latitude_count
        for index in range(longitude_count):
            phi = 2 * np.pi * index / longitude_count
            vertices.append([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    south = len(vertices)
    vertices.append([0, 0, -1])
    def ring(level, index):
        return 1 + (level - 1) * longitude_count + index % longitude_count
    faces = []
    for index in range(longitude_count):
        faces.append((0, ring(1, index), ring(1, index + 1)))
        for level in range(1, latitude_count - 1):
            faces.append((ring(level, index), ring(level + 1, index),
                          ring(level + 1, index + 1), ring(level, index + 1)))
        faces.append((south, ring(latitude_count - 1, index + 1), ring(latitude_count - 1, index)))
    return np.array(vertices), faces


def _custom_mesh(shape):
    vertices = np.asarray(shape.get("vertices_nm"), dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
            or not np.isfinite(vertices).all()):
        raise ValueError("Custom meshes need at least four finite 3D vertices_nm")
    if len(np.unique(vertices, axis=0)) != len(vertices):
        raise ValueError("Custom mesh vertices_nm must be distinct")
    # Preserve the user's local origin, which locates the particle on its SL site.
    scale = np.ptp(vertices, axis=0).max()
    normalized = (vertices - vertices.mean(axis=0)) / scale
    raw_faces = shape.get("faces")
    if not isinstance(raw_faces, (list, tuple)) or len(raw_faces) < 4:
        raise ValueError("A closed custom mesh requires at least four faces")
    faces = []
    edges = Counter()
    volume = 0.
    for raw in raw_faces:
        if (not isinstance(raw, (list, tuple)) or len(raw) < 3
                or any(type(i) is not int or i < 0 or i >= len(vertices) for i in raw)
                or len(set(raw)) != len(raw)):
            raise ValueError("Each face needs three or more distinct zero-based vertex indices")
        face = tuple(raw)
        points = normalized[list(face)]
        sides = np.roll(points, -1, axis=0) - points
        normal = np.sum(np.cross(points, sides), axis=0)
        area = np.linalg.norm(normal)
        if area < 1e-13:
            raise ValueError("Custom mesh contains a zero-area face")
        normal /= area
        if np.max(np.abs((points-points[0]) @ normal)) > 1e-8:
            raise ValueError("Custom polygon faces must be planar; use triangles for curved surfaces")
        for i, side in enumerate(sides):
            if np.min(np.cross(side, points-points[i]) @ normal) < -1e-10:
                raise ValueError("Custom polygon faces must be convex and cyclically ordered")
            edges[(face[i], face[(i+1) % len(face)])] += 1
        volume += (points[0] @ normal) * area / 6
        faces.append(face)
    if any(n != 1 or edges[(b, a)] != 1 for (a,b), n in edges.items()):
        raise ValueError("A closed custom mesh needs two oppositely wound faces at every edge")
    if set(i for face in faces for i in face) != set(range(len(vertices))):
        raise ValueError("Every custom vertex must belong to a face")
    if abs(volume) < 1e-13:
        raise ValueError("Custom mesh must enclose a nonzero volume")
    if volume < 0:
        faces = [face[::-1] for face in faces]
    return vertices, faces


def particle_mesh(shape):
    """Return local NC Cartesian vertices in nm and outward-wound polygon faces.

    Built-ins take type and radius_nm (center-to-vertex distance). The
    hexagonal prism has its long symmetry axis on local Cartesian z and
    all edges equal. For custom meshes supply zero-based faces and vertices_nm
    relative to the intended NC origin; use convex polygons or triangles.
    """
    if not isinstance(shape, dict):
        raise ValueError("Particle shape must be an object")
    kind = shape.get("type")
    if kind == "custom":
        return _custom_mesh(shape)
    radius = _array(shape.get("radius_nm"), (), "shape.radius_nm").item()
    if radius <= 0:
        raise ValueError("shape.radius_nm must be positive")
    axial = np.vstack((np.eye(3), -np.eye(3)))
    diagonals = np.array(list(product((-1, 1), repeat=3)))
    if kind == "sphere":
        vertices, faces = _sphere()
    elif kind == "hexagonal_prism":
        angles = np.pi / 3 * np.arange(6)
        vertices = np.array([[np.cos(a), np.sin(a), z] for z in (-.5, .5) for a in angles])
        faces = [tuple(reversed(range(6))), tuple(range(6, 12))]
        for i in range(6):
            j = (i+1) % 6
            faces.append((i, j, j+6, i+6))
    elif kind in ("cube", "octahedron", "cuboctahedron"):
        if kind == "cube":
            vertices, normals = diagonals, axial
        elif kind == "octahedron":
            vertices, normals = axial, diagonals
        else:
            vertices = np.array(sorted({p for x in (-1, 1) for y in (-1, 1)
                                        for p in permutations((x, y, 0))}))
            normals = np.vstack((axial, diagonals))
        faces = _support_faces(vertices, normals)
    else:
        raise ValueError(f"Unknown particle shape {kind!r}")
    return vertices * (radius / np.max(np.linalg.norm(vertices, axis=1))), faces
