"""Reference Debye scattering equation for finite atomistic assemblies.

    I(q) = sum_a w_a f_a(q)^2 + 2 sum_{a<b} w_a w_b f_a(q) f_b(q) sinc(q r_ab)

with sinc(x) = sin(x)/x (1 at x = 0), occupancy weights w and the same
Waasmaier-Kirfel f0 and isotropic B model as waxs.py (no anomalous terms,
so f is real). Pairs within and between particles are included, so a full
assembly keeps interparticle interference at wide angles. The result is a
powder (orientational) average; absolute laboratory orientation is lost but
pair distances are retained.

This NumPy implementation is the in-repository reference for small systems
and for checking external codes (AES-Debye, DebUsSy, DebyeCalculator, ...).
``bin_width_nm`` switches to a pair-distance histogram, which is much faster
for >~5000 atoms at the cost of a phase error of about q*bin/2.
"""

import numpy as np

from waxs import _atomic_factor


def _types(species, occupancy, b_iso):
    keys = list(zip(species, np.round(b_iso, 12)))
    unique = sorted(set(keys))
    index = np.array([unique.index(k) for k in keys])
    return unique, index


def debye_intensity(positions, species, q, *, occupancy=None, b_iso_nm2=None,
                    form_factors=None, bin_width_nm=None, chunk=None):
    """Powder intensity in electron units squared at q (rad/nm)."""
    x = np.asarray(positions, float)
    q = np.asarray(q, float)
    n = len(x)
    w = np.ones(n) if occupancy is None else np.asarray(occupancy, float)
    b = np.zeros(n) if b_iso_nm2 is None else np.asarray(b_iso_nm2, float)
    unique, kind = _types(species, w, b)
    # Keep each distance block near 2e7 entries (~160 MB) regardless of n.
    chunk = chunk or max(1, min(1024, 20_000_000 // max(n, 1)))
    factors = np.array([_atomic_factor(sym, q, form_factors or {}) * np.exp(-bb * (q / (4*np.pi))**2)
                        for sym, bb in unique])  # (types, nq)
    self_term = np.zeros_like(q)
    for t in range(len(unique)):
        self_term += w[kind == t].sum() * factors[t] ** 2
    cross = np.zeros_like(q)
    if bin_width_nm:
        r_max = np.sqrt(3) * np.ptp(x, axis=0).max() + bin_width_nm if n else bin_width_nm
        edges = np.arange(0, r_max + 2 * bin_width_nm, bin_width_nm)
        centers = 0.5 * (edges[1:] + edges[:-1])
        hist = np.zeros((len(unique), len(unique), len(centers)))
        for s in range(0, n, chunk):
            d = np.linalg.norm(x[s:s + chunk, None] - x[None], axis=-1)
            weight = w[s:s + chunk, None] * w[None]
            upper = np.arange(s, min(s + chunk, n))[:, None] < np.arange(n)[None]
            for ta in range(len(unique)):
                rows = kind[s:s + chunk] == ta
                for tb in range(len(unique)):
                    m = upper & rows[:, None] & (kind == tb)[None]
                    if m.any():
                        hist[ta, tb] += np.histogram(d[m], edges, weights=weight[m])[0]
        sinc = np.sinc(np.outer(q, centers) / np.pi)  # (nq, bins)
        for ta in range(len(unique)):
            for tb in range(len(unique)):
                if hist[ta, tb].any():
                    cross += factors[ta] * factors[tb] * (sinc @ hist[ta, tb])
    else:
        f_atom = factors[kind] * w[:, None]  # (n, nq)
        for s in range(0, n, chunk):
            d = np.linalg.norm(x[s:s + chunk, None] - x[None], axis=-1)
            d = np.where(np.arange(s, min(s + chunk, n))[:, None] < np.arange(n)[None], d, np.nan)
            rows, cols = np.nonzero(np.isfinite(d))
            for p in range(0, len(rows), 20_000):
                r, c = rows[p:p + 20_000], cols[p:p + 20_000]
                arg = np.outer(d[r, c], q)
                cross += np.sum(f_atom[s + r] * f_atom[c] * np.sinc(arg / np.pi), axis=0)
    return self_term + 2 * cross


def assembly_intensity(assembly, q, **kwargs):
    return debye_intensity(assembly["positions"], assembly["species"], q,
                           occupancy=assembly["occupancy"], b_iso_nm2=assembly["b_iso_nm2"], **kwargs)


def isolated_sum(assembly, q, **kwargs):
    """Sum of single-particle Debye curves: the incoherent-addition limit."""
    total = np.zeros_like(np.asarray(q, float))
    for j in np.unique(assembly["particle"]):
        m = assembly["particle"] == j
        total += debye_intensity(assembly["positions"][m], [s for s, k in zip(assembly["species"], m) if k],
                                 q, occupancy=assembly["occupancy"][m],
                                 b_iso_nm2=assembly["b_iso_nm2"][m], **kwargs)
    return total
