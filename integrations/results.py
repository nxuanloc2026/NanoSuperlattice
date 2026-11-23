"""Import external or measured scattering results with their metadata.

A dataset is a CSV file plus a run.json sidecar written by
``exchange.run_metadata`` (or by hand). The importer converts q to rad/nm,
keeps raw intensity and uncertainty, and records backend, version, axes,
units and normalization. Missing metadata is reported in ``warnings``.

Accepted CSV layouts (header row required; lines starting with # are skipped):
  profile  q[unit], intensity[, uncertainty]       or 2theta_deg instead of q
  image    pixel_x, pixel_y, qx, qy[, qz], intensity[, uncertainty]
Column names are matched case-insensitively. A unit in the header, such as
``q_A^-1`` or ``q (1/nm)``, overrides ``units.q`` in run.json.
"""

import csv
import json
from pathlib import Path
import re

import numpy as np

from .conventions import to_q_nm_inv

REGIMES = ("saxs", "waxs", "detector")
_INTENSITY = ("intensity", "i", "intensity_au", "intensity_raw", "counts", "i(q)")
_SIGMA = ("uncertainty", "sigma", "error", "err", "di", "esd", "std")


def _sidecar(csv_path):
    path = Path(csv_path)
    for candidate in (path.with_suffix(".run.json"), path.with_name(path.stem + "_run.json"),
                      path.with_name("run.json")):
        if candidate.exists():
            return candidate
    return None


def _split_header(name):
    """'q (1/A)' -> ('q', '1/a'); 'q_nm^-1' -> ('q', 'nm^-1')."""
    text = name.strip().lower()
    match = re.fullmatch(r"([a-z0-9()_]+?)\s*[\[(]\s*(.+?)\s*[\])]", text)
    if match:
        return match.group(1), match.group(2)
    for prefix in ("qx", "qy", "qz", "q"):
        if text.startswith(prefix + "_") and len(text) > len(prefix) + 1:
            return prefix, text[len(prefix) + 1:]
    return text, None


def _read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        lines = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.reader(lines)
    header = next(reader)
    rows = [[float(v) if v.strip() not in ("", "nan", "NaN") else np.nan for v in row]
            for row in reader]
    return [h.strip() for h in header], np.asarray(rows, dtype=float)


def read_dataset(csv_path, run_json=None, *, name=None, regime=None):
    path = Path(csv_path)
    meta_path = Path(run_json) if run_json else _sidecar(path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path else {}
    warnings = [] if meta_path else ["No run.json metadata found; units and provenance are assumed."]
    header, data = _read_rows(path)
    if data.ndim != 2 or not len(data):
        raise ValueError(f"{path.name} has no numeric rows.")
    columns = {}
    for i, raw in enumerate(header):
        base, unit = _split_header(raw)
        columns[base] = (i, unit)
    units = meta.get("units", {})
    wavelength = meta.get("wavelength_nm")
    for key in ("backend", "normalization"):
        if not meta.get(key):
            warnings.append(f"run.json does not state '{key}'.")

    def column(*names):
        for n in names:
            if n in columns:
                return columns[n]
        return None

    intensity_col = column(*_INTENSITY)
    if intensity_col is None:
        raise ValueError(f"{path.name} needs an intensity column ({', '.join(_INTENSITY)}).")
    sigma_col = column(*_SIGMA)
    dataset = {"name": name or meta.get("name") or path.stem, "source": path.name,
               "backend": meta.get("backend", "unknown"),
               "backend_version": meta.get("backend_version"),
               "normalization": meta.get("normalization", "unknown"),
               "intensity_unit": units.get("intensity", "unknown"),
               "wavelength_nm": wavelength, "axes": meta.get("axes", {}),
               "geometry": meta.get("geometry"), "metadata": meta, "warnings": warnings}
    intensity = data[:, intensity_col[0]]
    sigma = data[:, sigma_col[0]] if sigma_col else None
    if sigma is not None and not np.isfinite(sigma).any():
        sigma = None  # an empty uncertainty column means "not measured"

    if column("qx") and column("qy"):
        qcols = [column(k) for k in ("qx", "qy", "qz")]
        q = np.column_stack([to_q_nm_inv(data[:, c[0]], c[1] or units.get("q", "nm^-1"), wavelength)
                             if c else np.zeros(len(data)) for c in qcols])
        px, py = column("pixel_x", "px", "x"), column("pixel_y", "py", "y")
        if not (px and py):
            raise ValueError("Detector data need pixel_x and pixel_y columns.")
        ix, iy = data[:, px[0]].astype(int), data[:, py[0]].astype(int)
        nx, ny = ix.max() + 1, iy.max() + 1
        image = np.full((ny, nx), np.nan)
        image[iy, ix] = intensity
        grids = [np.full((ny, nx), np.nan) for _ in range(3)]
        for g, values in zip(grids, q.T):
            g[iy, ix] = values
        dataset.update(kind="image", regime=regime or meta.get("regime") or "detector",
                       image=image, qx=grids[0], qy=grids[1], qz=grids[2],
                       uncertainty=None if sigma is None else _grid(sigma, ix, iy, nx, ny))
        if not dataset["axes"]:
            warnings.append("Detector axes are not described in run.json.")
        return dataset

    qcol = column("q", "|q|", "q_nm", "s")
    if qcol is not None:
        unit = qcol[1] or units.get("q")
        if unit is None:
            warnings.append("q unit not given; assuming rad/nm.")
            unit = "nm^-1"
        q = to_q_nm_inv(data[:, qcol[0]], unit, wavelength)
    else:
        angle = column("2theta", "two_theta", "2theta_deg", "two_theta_deg", "tth")
        if angle is None:
            raise ValueError(f"{path.name} needs a q or 2theta column.")
        q = to_q_nm_inv(data[:, angle[0]], "2theta_rad" if angle[1] == "rad" else "2theta_deg", wavelength)
    order = np.argsort(q)
    found = regime or meta.get("regime")
    if found not in REGIMES:
        found = "waxs" if np.nanmax(q) > 8 else "saxs"
        warnings.append(f"Regime not stated; classified as {found.upper()} from the q range.")
    dataset.update(kind="profile", regime=found, q=q[order], intensity=intensity[order],
                   uncertainty=None if sigma is None else sigma[order])
    return dataset


def _grid(values, ix, iy, nx, ny):
    grid = np.full((ny, nx), np.nan)
    grid[iy, ix] = values
    return grid


def _finite_list(array):
    return [None if not np.isfinite(v) else float(v) for v in np.ravel(array)]


def viewer_payload(dataset, max_pixels=256):
    """JSON-safe, size-limited form of a dataset for the HTML viewer."""
    out = {k: dataset[k] for k in ("name", "source", "backend", "backend_version", "regime",
                                   "kind", "normalization", "intensity_unit", "warnings")}
    out["wavelengthNm"] = dataset.get("wavelength_nm")
    if dataset["kind"] == "profile":
        out.update(q=_finite_list(dataset["q"]), intensity=_finite_list(dataset["intensity"]),
                   uncertainty=None if dataset["uncertainty"] is None
                   else _finite_list(dataset["uncertainty"]))
        return out
    image = dataset["image"]
    step = int(np.ceil(max(image.shape) / max_pixels))
    if step > 1:
        out["warnings"] = out["warnings"] + [f"Displayed every {step}th pixel; CSV keeps all values."]
    pick = (slice(None, None, step), slice(None, None, step))
    out.update(image=[_finite_list(r) for r in image[pick]],
               qx=[_finite_list(r) for r in dataset["qx"][pick]],
               qy=[_finite_list(r) for r in dataset["qy"][pick]],
               qz=[_finite_list(r) for r in dataset["qz"][pick]], pixelStep=step)
    return out


def profile_peaks(q, intensity, threshold=0.05):
    """Interior local maxima above threshold * max, refined by a parabola."""
    q, y = np.asarray(q, float), np.asarray(intensity, float)
    top = np.nanmax(y)
    peaks = []
    for i in np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1:
        if y[i] < threshold * top:
            continue
        den = y[i - 1] - 2 * y[i] + y[i + 1]
        offset = 0.5 * (y[i - 1] - y[i + 1]) / den if den else 0.0
        peaks.append(float(q[i] + offset * (q[i + 1] - q[i - 1]) / 2))
    return peaks


def compare_profiles(a, b, *, threshold=0.05, floor=1e-5, tolerance=0.03):
    """Match q positions first, then compare normalized shapes on the shared q range.

    a and b are profile datasets (dicts with q and intensity). Each curve is
    scaled to its own maximum inside the common range before comparison.
    Peaks match when |q_a - q_b| <= tolerance * q_a; the rest are listed as
    unmatched (finite-size fringes, interparticle maxima, missing reflections).
    """
    lo = max(np.nanmin(a["q"]), np.nanmin(b["q"]))
    hi = min(np.nanmax(a["q"]), np.nanmax(b["q"]))
    if not hi > lo:
        raise ValueError("The two profiles share no q range.")
    grid = np.linspace(lo, hi, 2000)
    ya = np.interp(grid, a["q"], a["intensity"])
    yb = np.interp(grid, b["q"], b["intensity"])
    ya, yb = ya / ya.max(), yb / yb.max()
    pa, pb = profile_peaks(grid, ya, threshold), profile_peaks(grid, yb, threshold)
    matches, used = [], set()
    for p in pa:
        candidates = [(abs(v - p), k) for k, v in enumerate(pb)
                      if k not in used and abs(v - p) <= tolerance * p]
        if candidates:
            k = min(candidates)[1]
            used.add(k)
            matches.append({"q_a": p, "q_b": pb[k], "delta_q": pb[k] - p})
    matched_a = {m["q_a"] for m in matches}
    la, lb = np.log10(np.maximum(ya, floor)), np.log10(np.maximum(yb, floor))
    return {"q_range_nm_inv": [float(lo), float(hi)], "peaks_a": pa, "peaks_b": pb,
            "matches": matches, "unmatched_a": [p for p in pa if p not in matched_a],
            "unmatched_b": [v for k, v in enumerate(pb) if k not in used],
            "max_peak_shift_nm_inv": max((abs(m["delta_q"]) for m in matches), default=None),
            "log_correlation": float(np.corrcoef(la, lb)[0, 1]),
            "rms_difference_normalized": float(np.sqrt(np.mean((ya - yb) ** 2)))}


def parse_overlay_argument(text):
    """'data.csv' or 'data.csv,run.json' from the command line."""
    parts = [p for p in text.split(",") if p]
    if not 1 <= len(parts) <= 2:
        raise ValueError("--overlay expects CSV or CSV,RUN_JSON")
    return read_dataset(parts[0], parts[1] if len(parts) == 2 else None)
