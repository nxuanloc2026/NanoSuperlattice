"""Kinematic SAXS of the finite particle assembly, in the superlattice frame.

The amplitude is the Fourier transform (positive phase convention) of the
particle/medium excess density. Lengths are nm and q includes the 2*pi factor
(nm^-1). Absolute intensity here means the unnormalized squared amplitude,
without flux, cross-section, polarization, beamstop or detector corrections.
"""

from typing import NamedTuple

import numpy as np


def _sphere_amplitude(q, radius):
    x = radius * np.linalg.norm(q, axis=1)
    small = x < 0.05
    form = np.empty(len(x))
    y = x[small] ** 2
    form[small] = 1 - y / 10 + y**2 / 280 - y**3 / 15120
    t = x[~small]
    form[~small] = 3 * (np.sin(t) - t * np.cos(t)) / t**3
    return (4 * np.pi / 3 * radius**3) * form


class _Polyhedron(NamedTuple):
    vertices: np.ndarray
    triangles: np.ndarray
    area_vectors: np.ndarray
    origin: np.ndarray
    centroid: np.ndarray
    radius: float
    volume: float


def _prepare_mesh(mesh):
    # Moving the expansion origin close to the particle keeps phases well
    # conditioned for an off-center custom mesh. Restore the shift below.
    vertices = np.asarray(mesh["vertices"], dtype=float)
    origin = vertices.mean(axis=0)
    vertices = vertices - origin
    triangles = np.array([(face[0], face[j], face[j+1])
                          for face in mesh["faces"] for j in range(1, len(face)-1)])
    a, b, c = vertices[triangles].transpose(1, 0, 2)
    area_vectors = np.cross(b - a, c - a) / 2
    signed_volume = np.einsum("ij,ij->i", a, area_vectors) / 3
    volume = signed_volume.sum()
    centroid = signed_volume @ ((a + b + c) / 4) / volume
    return _Polyhedron(vertices, triangles, area_vectors, origin, centroid,
                       np.linalg.norm(vertices, axis=1).max(), volume)


def _triangle_mean(phases):
    """Mean exp(i q.r) on triangles, given the phases at their three vertices.

    The analytic integral is minus twice the second divided difference of
    exp(i*t). A centered Taylor series handles nearly equal vertex phases.
    """
    low, mid, high = np.sort(phases, axis=-1).transpose(2, 0, 1)
    span = high - low
    wide = span >= 0.03
    result = np.empty_like(span, dtype=complex)
    a, b, c = low[wide], mid[wide], high[wide]
    # np.sinc(x) = sin(pi*x)/(pi*x); these divided differences stay finite
    # even when two corner phases coincide on a symmetry direction.
    left = np.exp(0.5j * (a+b)) * np.sinc((b-a)/(2*np.pi))
    right = np.exp(0.5j * (b+c)) * np.sinc((c-b)/(2*np.pi))
    result[wide] = -2j * (right - left) / span[wide]
    a, b, c = low[~wide], mid[~wide], high[~wide]
    mean = (a + b + c) / 3
    a, b, c = a-mean, b-mean, c-mean
    e2, e3 = a*b + a*c + b*c, a*b*c
    result[~wide] = np.exp(1j * mean) * (
        1 + e2/12 - 1j*e3/60 + e2**2/360 - 1j*e2*e3/1260)
    return result


def _polyhedron_amplitude(q, mesh):
    # Divergence theorem reduces the volume FT to triangle surface integrals.
    # Background: J. Wuttke (2021), doi:10.1107/S1600576721001710.
    q2 = np.einsum("ij,ij->i", q, q)
    wide = q2 * mesh.radius**2 >= 1e-12
    result = np.empty(len(q), dtype=complex)
    p = q[wide]
    phases = np.einsum("ij,kj->ik", p, mesh.vertices)
    normals = np.einsum("ij,kj->ik", p, mesh.area_vectors)
    face_mean = _triangle_mean(phases[:, mesh.triangles])
    # The oriented area vectors of a closed surface sum to zero.
    result[wide] = np.sum(normals * (face_mean - 1), axis=1) / (1j*q2[wide])
    result[~wide] = mesh.volume * np.exp(1j * (q[~wide] @ mesh.centroid))
    return result * np.exp(1j * (q @ mesh.origin))


def _assembly_intensity(scene, config, q, contrast):
    """Coherently sum shape amplitudes times finite sums of particle phases."""
    amplitude = np.zeros(len(q), dtype=complex)
    for i, variant in enumerate(scene["variants"]):
        centers = np.array([s["xyz"] for s in scene["centers"] if s["variant"] == i])
        if not len(centers) or contrast[i] == 0:
            continue
        shape = config["variants"][i].get("shape") or config["shape"]
        sphere = shape["type"] == "sphere"
        mesh = None if sphere else _prepare_mesh(variant["mesh"])
        rotation = np.asarray(variant["rotation"])
        # Bound intermediate arrays even for large custom meshes/site counts.
        block = 2048 if sphere else max(1, min(2048, 100000 // len(mesh.triangles)))
        for start in range(0, len(q), block):
            vectors = q[start:start+block]
            form = (_sphere_amplitude(vectors, shape["radius_nm"]) if sphere else
                    _polyhedron_amplitude(vectors @ rotation, mesh))
            lattice_sum = np.zeros(len(vectors), dtype=complex)
            for chunk in range(0, len(centers), 128):
                phase = np.einsum("ij,kj->ik", vectors, centers[chunk:chunk+128])
                lattice_sum += np.sum(np.exp(1j * phase), axis=1)
            amplitude[start:start+block] += contrast[i] * form * lattice_sum
    return np.abs(amplitude)**2


def simulate_saxs(scene, config, q_axis, ex, ey, radial_q, settings):
    """Return a central reciprocal-space intensity slice and its radial profile.

    q = qx*ex + qy*ey, with orthonormal in-plane Cartesian SL axes. Image
    indices are [iy, ix]. The profile evaluates the same model at radial_q
    with uniform azimuthal quadrature. azimuth_samples (default 128) covers
    half a circle because real contrast gives I(-q) = I(q). Increase it to
    converge sharp angular features of a large assembly. This is an in-plane
    average, distinct from a 3D powder average.

    settings is the scattering.saxs object. contrast is a real density
    difference shared by all variants or a list in variant order (default 1).
    For electron-density contrast in e/nm^3, the returned values are in e^2.
    I(0) = (sum_j contrast_j * volume_j)^2, including for matched contrast.
    """
    q_axis, radial_q = np.asarray(q_axis, dtype=float), np.asarray(radial_q, dtype=float)
    if (q_axis.ndim != 1 or radial_q.ndim != 1 or np.any(radial_q < 0)
            or not np.isfinite(q_axis).all() or not np.isfinite(radial_q).all()):
        raise ValueError("SAXS needs finite 1D q coordinates and nonnegative radial q.")
    plane = np.asarray([ex, ey], dtype=float)
    if plane.shape != (2, 3) or not np.allclose(plane @ plane.T, np.eye(2), atol=1e-7, rtol=0):
        raise ValueError("SAXS slice axes ex and ey must be orthonormal Cartesian vectors.")
    contrast = np.asarray(settings.get("contrast", 1), dtype=float)
    count = len(scene["variants"])
    if contrast.ndim == 0:
        contrast = np.repeat(contrast, count)
    if contrast.shape != (count,) or not np.isfinite(contrast).all():
        raise ValueError("SAXS contrast must be finite and scalar or one real value per variant.")
    samples = settings.get("azimuth_samples", 128)
    if type(samples) is not int or samples < 2:
        raise ValueError("SAXS azimuth_samples must be an integer of at least 2.")
    pixels = q_axis[None, :, None] * plane[0] + q_axis[:, None, None] * plane[1]
    phi = np.arange(samples) * (np.pi / samples)
    rays = np.cos(phi)[:, None] * plane[0] + np.sin(phi)[:, None] * plane[1]
    rings = radial_q[:, None, None] * rays
    vectors = np.concatenate((pixels.reshape(-1, 3), rings.reshape(-1, 3)))
    result = _assembly_intensity(scene, config, vectors, contrast)
    split = len(q_axis)**2
    return (result[:split].reshape(len(q_axis), len(q_axis)),
            result[split:].reshape(len(radial_q), samples).mean(axis=1))
