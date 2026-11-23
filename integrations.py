"""Optional bridges between NanoSuperlattice and external scattering software.

One module, run from the repository root:

    python integrations.py --help
    python integrations.py export examples/silicon_fcc.json exchange/si

Sections, in order: conventions, exchange, symmetry, results, dans, bornagain,
atomistic, debye, scattersim, formats, probe, cli.

Nothing here is imported by the core viewer. External packages (BornAgain,
Dans_Diffraction) are imported lazily inside the function that needs them,
so the core keeps NumPy as its only dependency.

Shared conventions (see the conventions section):
  * lengths in nm, angles in degrees, q in rad/nm (q = 2*pi/d);
  * cell matrices A hold direct basis vectors in rows, x = f @ A;
  * particle rotations are active, x_SL = R @ x_NC;
  * a separate laboratory rotation maps the SL frame to the lab, x_lab = U @ x_SL.
"""

import argparse
import copy
import csv
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import pkgutil
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

from crystal_geometry import index_vector, lattice_matrix, particle_mesh
from saxs import _assembly_intensity
from visualize_superlattice import build_scene
from waxs import _atomic_factor, _basis_sites

FORMAT_VERSION = "1.0"

# ============================================================================
# conventions
# ============================================================================
# Units, frames and vector conventions shared by every exchange file.
#
# A holds the direct lattice vectors in rows (nm). Then
#   fractional row f      -> Cartesian x = f @ A
#   direction [uvw]       -> A.T @ [u, v, w]
#   plane normal (hkl)    -> inv(A) @ [h, k, l]
#   reciprocal vector G   -> 2*pi * inv(A) @ [h, k, l]       (rad/nm)
# Particle rotations R are active (x_SL = R @ x_NC, G_SL = R @ G_NC). A
# separate proper rotation U maps SL Cartesian coordinates into the laboratory
# frame (x_lab = U @ x_SL). U is never folded into a particle rotation.

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


# ============================================================================
# exchange
# ============================================================================
# Exchange files: structure.cif, particles.csv, orientations.json, run.json.
#
#     from integrations import export_bundle, round_trip_check
#     export_bundle(config, "exchange/silicon")
#     print(round_trip_check(config, "exchange/silicon"))
#
# CIF files are written in angstrom with a symmetry-expanded P1 basis. All other
# files keep nm and rad/nm. The CIF reader also expands simple space-group
# operators (x,y,z strings) so structures from databases can be imported.

ROTATION_CONVENTION = ("active rotation acting on Cartesian column vectors: "
                       "x_SL = R @ x_NC; for row arrays x_SL = x_NC @ R.T")
LAB_CONVENTION = "x_lab = U @ x_SL; U is independent of every particle rotation"
HERE = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------- CIF
def cif_text(atomic, title="NanoSuperlattice structure"):
    """Return a P1 CIF for the unique atomic sites (cell in angstrom)."""
    cell = np.asarray(atomic["cell"], dtype=float)
    lines = [f"data_{re.sub(r'[^A-Za-z0-9_]+', '_', title)[:60] or 'structure'}",
             f"_audit_creation_method 'NanoSuperlattice integrations {FORMAT_VERSION}'",
             f"_chemical_name_common {_quote(title)}"]
    for key, value in zip(("a", "b", "c"), cell[:3] * 10):
        lines.append(f"_cell_length_{key} {value:.8f}")
    for key, value in zip(("alpha", "beta", "gamma"), cell[3:]):
        lines.append(f"_cell_angle_{key} {value:.8f}")
    lines += ["_symmetry_space_group_name_H-M 'P 1'", "_space_group_IT_number 1", "",
              "loop_", "_space_group_symop_operation_xyz", "'x, y, z'", "",
              "loop_"] + [f"_atom_site_{k}" for k in (
                  "label", "type_symbol", "fract_x", "fract_y", "fract_z",
                  "occupancy", "U_iso_or_equiv", "adp_type")]
    counts = {}
    for symbol, position, occupancy, b_iso in _basis_sites(atomic):
        element = re.match(r"[A-Za-z]+", symbol).group(0)
        counts[element] = counts.get(element, 0) + 1
        # B (nm^2) -> B (A^2) = 100*B; U = B / (8*pi^2).
        u_iso = 100 * b_iso / (8 * np.pi**2)
        x, y, z = (f"{v:.10f}" for v in position)
        lines.append(f"{element}{counts[element]} {_quote(symbol)} {x} {y} {z} "
                     f"{occupancy:.6f} {u_iso:.10f} Uiso")
    return "\n".join(lines) + "\n"


def _quote(text):
    text = str(text)
    if re.fullmatch(r"[^\s'\"#$_;\[\]][^\s]*", text):
        return text
    return "'" + text.replace("'", "") + "'"


# CIF 1.1: a quote closes a string only when followed by whitespace.
_TOKEN = re.compile(r"'((?:[^']|'(?=\S))*)'(?=\s|$)|\"((?:[^\"]|\"(?=\S))*)\"(?=\s|$)|(#.*)|(\S+)")


def _cif_tokens(text):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            block = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                block.append(lines[i])
                i += 1
            yield "\n".join(block).strip()
        else:
            for single, double, comment, bare in _TOKEN.findall(line):
                if comment:
                    break
                yield bare or single or double
        i += 1


def parse_cif(text):
    """Return (items, loops) for the first data block of a CIF string."""
    items, loops, tokens = {}, [], list(_cif_tokens(text))
    i, started = 0, False
    while i < len(tokens):
        token = tokens[i]
        low = token.lower()
        if low.startswith("data_"):
            if started:
                break
            started = True
            i += 1
        elif low == "loop_":
            names = []
            i += 1
            while i < len(tokens) and tokens[i].startswith("_"):
                names.append(tokens[i].lower())
                i += 1
            values = []
            while i < len(tokens) and not tokens[i].startswith("_") \
                    and tokens[i].lower() not in ("loop_",) and not tokens[i].lower().startswith("data_"):
                values.append(tokens[i])
                i += 1
            if names and len(values) % len(names):
                raise ValueError(f"CIF loop {names[0]} has an incomplete row.")
            rows = [values[j:j + len(names)] for j in range(0, len(values), len(names))]
            loops.append({name: [row[k] for row in rows] for k, name in enumerate(names)})
        elif token.startswith("_"):
            items[low] = tokens[i + 1] if i + 1 < len(tokens) else ""
            i += 2
        else:
            i += 1
    return items, loops


def _number(text):
    """CIF numbers may carry a standard uncertainty in parentheses."""
    if text in (".", "?"):
        return None
    return float(re.sub(r"\(\d+\)$", "", text))


def parse_symop(op):
    """Parse 'x,y,z' style operators to (W, w) with f' = W @ f + w."""
    parts = op.replace(" ", "").lower().split(",")
    if len(parts) != 3:
        raise ValueError(f"Symmetry operator {op!r} needs three components.")
    w_matrix, shift = np.zeros((3, 3)), np.zeros(3)
    for row, part in enumerate(parts):
        for sign, body in re.findall(r"([+-]?)([^+-]+)", part):
            factor = -1.0 if sign == "-" else 1.0
            axis = body[-1] if body[-1] in "xyz" else None
            coefficient = body[:-1].rstrip("*") if axis else body
            if coefficient:
                num, _, den = coefficient.partition("/")
                factor *= float(num) / float(den or 1)
            if axis:
                w_matrix[row, "xyz".index(axis)] += factor
            else:
                shift[row] += factor
    return w_matrix, shift


def atomic_from_cif(text, tolerance=1e-4):
    """Build a NanoSuperlattice 'atomic' block (nm) from CIF text."""
    items, loops = parse_cif(text)
    try:
        cell = [_number(items[f"_cell_length_{k}"]) / 10 for k in "abc"]
        cell += [_number(items[f"_cell_angle_{k}"]) for k in ("alpha", "beta", "gamma")]
    except KeyError as exc:
        raise ValueError(f"CIF is missing {exc.args[0]}") from exc
    ops = ["x,y,z"]
    for loop in loops:
        for key in ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"):
            if key in loop:
                ops = loop[key]
    operators = [parse_symop(op) for op in ops]
    site_loop = next((loop for loop in loops if "_atom_site_fract_x" in loop), None)
    if site_loop is None:
        raise ValueError("CIF has no _atom_site_fract_x loop.")
    n = len(site_loop["_atom_site_fract_x"])
    basis = []
    for i in range(n):
        label = site_loop.get("_atom_site_label", [f"X{i}"] * n)[i]
        symbol = site_loop.get("_atom_site_type_symbol", [None] * n)[i] or re.match(
            r"[A-Za-z]+", label).group(0)
        frac = np.array([_number(site_loop[f"_atom_site_fract_{k}"][i]) for k in "xyz"])
        occupancy = _number(site_loop.get("_atom_site_occupancy", ["1"] * n)[i])
        b_nm2 = 0.0
        if "_atom_site_u_iso_or_equiv" in site_loop:
            u = _number(site_loop["_atom_site_u_iso_or_equiv"][i])
            b_nm2 = 0.0 if u is None else 8 * np.pi**2 * u / 100
        elif "_atom_site_b_iso_or_equiv" in site_loop:
            b = _number(site_loop["_atom_site_b_iso_or_equiv"][i])
            b_nm2 = 0.0 if b is None else b / 100
        seen = []
        for w_matrix, shift in operators:
            position = np.mod(w_matrix @ frac + shift, 1.0)
            position[np.isclose(position, 1.0, atol=tolerance)] = 0.0
            if any(np.all(np.abs((position - p + 0.5) % 1.0 - 0.5) < tolerance) for p in seen):
                continue
            seen.append(position)
            site = {"element": symbol, "frac": [float(v) for v in position]}
            if occupancy is not None and occupancy != 1:
                site["occupancy"] = float(occupancy)
            if b_nm2:
                site["b_iso_nm2"] = float(b_nm2)
            basis.append(site)
    return {"cell": cell, "basis": basis}


# --------------------------------------------------------------- particles
def particle_rows(scene):
    names = [v["name"] for v in scene["variants"]]
    for i, center in enumerate(scene["centers"]):
        yield {"site_id": i, "cell_i": center["cell"][0], "cell_j": center["cell"][1],
               "cell_k": center["cell"][2], "basis_site": center["site"],
               "x_nm": center["xyz"][0], "y_nm": center["xyz"][1], "z_nm": center["xyz"][2],
               "variant_id": center["variant"], "variant_name": names[center["variant"]]}


def write_particles_csv(scene, path):
    rows = list(particle_rows(scene))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (repr(float(v)) if k.endswith("_nm") else v) for k, v in row.items()})
    return path


def read_particles_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    centers = np.array([[float(r[k]) for k in ("x_nm", "y_nm", "z_nm")] for r in rows])
    variants = np.array([int(r["variant_id"]) for r in rows])
    return centers, variants


# ------------------------------------------------------------ orientations
def orientation_document(config, scene):
    sl_cell = np.asarray(scene["slCell"])
    counts = np.bincount([c["variant"] for c in scene["centers"]],
                         minlength=len(scene["variants"]))
    lab = laboratory_matrix(config, sl_cell)
    return {
        "format": "NanoSuperlattice orientations", "format_version": FORMAT_VERSION,
        "convention": ROTATION_CONVENTION,
        "frames": {
            "NC": "nanocrystal local Cartesian frame: x || a, y in the ab plane, z || a x b",
            "SL": "superlattice Cartesian frame built the same way from the SL cell",
            "lab": "laboratory frame defined by the 'laboratory' block (identity if absent)"},
        "units": {"length": "nm"},
        "nc_cell_matrix_rows_nm": scene["ncCell"], "sl_cell_matrix_rows_nm": scene["slCell"],
        "laboratory": {"matrix": lab.tolist(), "convention": LAB_CONVENTION},
        "variants": [{"id": i, "name": v["name"], "matrix": v["rotation"],
                      "site_count": int(counts[i]),
                      "population_fraction": float(counts[i] / counts.sum()),
                      "alignments": [{"nc": a["nc"], "sl": a["sl"]} for a in v["alignments"]]}
                     for i, v in enumerate(scene["variants"])],
    }


def read_orientations(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    rotations = [check_rotation(v["matrix"], f"variant {v['id']}") for v in doc["variants"]]
    lab = check_rotation(doc["laboratory"]["matrix"], "laboratory matrix")
    return rotations, lab, doc


# --------------------------------------------------------------- run.json
def _git_version():
    try:
        out = subprocess.run(["git", "-C", str(HERE), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run_metadata(config, *, backend="NanoSuperlattice", backend_version=None,
                 regime=None, q_unit="nm^-1", intensity_unit="model units (|amplitude|^2)",
                 normalization="raw model intensity; no display normalization",
                 axes=None, settings=None, files=None, extra=None):
    """Describe one calculation or measurement so it can be reproduced and compared."""
    options = config.get("scattering", {}) if isinstance(config, dict) else {}
    wavelength = float(options.get("wavelength_nm", 0.15406))
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    meta = {"format": "NanoSuperlattice run", "format_version": FORMAT_VERSION,
            "backend": backend,
            "backend_version": backend_version or (_git_version() if backend == "NanoSuperlattice" else None),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_sha256": hashlib.sha256(canonical).hexdigest(),
            "regime": regime, "wavelength_nm": wavelength, "energy_kev": energy_kev(wavelength),
            "units": {"length": "nm", "angle": "deg", "q": q_unit,
                      "q_convention": "q = 4*pi*sin(theta)/lambda = 2*pi/d",
                      "intensity": intensity_unit},
            "normalization": normalization, "axes": axes or {},
            "settings": settings if settings is not None else options, "files": files or []}
    if extra:
        meta.update(extra)
    return meta


def write_profile_csv(path, q, intensity, uncertainty=None, header=("q_nm^-1", "intensity", "uncertainty")):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i, (qv, iv) in enumerate(zip(q, intensity)):
            sigma = "" if uncertainty is None else repr(float(uncertainty[i]))
            writer.writerow([repr(float(qv)), repr(float(iv)), sigma])
    return path


# ----------------------------------------------------------------- bundle
def export_bundle(config, directory, *, scattering=None):
    """Write the complete exchange set for one configuration. Returns paths."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    scene = build_scene(config)
    paths = {"config": out / "config.json", "structure": out / "structure.cif",
             "particles": out / "particles.csv", "orientations": out / "orientations.json"}
    paths["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")
    paths["structure"].write_text(cif_text(config["atomic"], config.get("title", "structure")),
                                  encoding="utf-8")
    write_particles_csv(scene, paths["particles"])
    paths["orientations"].write_text(json.dumps(orientation_document(config, scene), indent=2),
                                     encoding="utf-8")
    axes = {}
    if scattering is None:
        scattering = "scattering" in config
    if scattering:
        from scattering import simulate_scattering
        result = simulate_scattering(config, scene)
        axes = {k: result[k] for k in ("xCartesian", "yCartesian", "normalCartesian")}
        axes["frame"] = "SL Cartesian; map pixel q = qx*xCartesian + qy*yCartesian"
        wavelength = result["wavelengthNm"]
        for regime in ("saxs", "waxs"):
            key = f"scattering_{regime}"
            paths[key] = write_profile_csv(out / f"{key}.csv", result[regime]["q"],
                                           result[regime]["intensity"])
            # Per-file sidecar so the importer knows the regime and averaging.
            averaging = ("azimuthal mean in the central slice" if regime == "saxs"
                         else "3D powder orientational mean, Gaussian-broadened")
            sidecar = run_metadata(config, regime=regime, axes=axes, files=[paths[key].name],
                                   extra={"name": f"NanoSuperlattice {regime.upper()}",
                                          "averaging": averaging})
            paths[key + "_run"] = out / f"{key}.run.json"
            paths[key + "_run"].write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        paths["reflections"] = out / "waxs_reflections.csv"
        with open(paths["reflections"], "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["h", "k", "l", "q_nm^-1", "d_nm", "two_theta_deg",
                             "multiplicity", "sum_F2"])
            for p in result["waxs"]["peaks"]:
                writer.writerow([*p["hkl"], p["q"], p["d"], float(two_theta_deg(p["q"], wavelength)),
                                 p["multiplicity"], p["intensity"]])
    paths["run"] = out / "run.json"
    meta = run_metadata(config, regime="saxs+waxs" if scattering else "geometry", axes=axes,
                        files=sorted(p.name for p in paths.values()))
    paths["run"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths


def load_bundle(directory):
    """Reload geometry from an exchange directory without using config.json."""
    out = Path(directory)
    atomic = atomic_from_cif((out / "structure.cif").read_text(encoding="utf-8"))
    centers, variant_ids = read_particles_csv(out / "particles.csv")
    rotations, lab, doc = read_orientations(out / "orientations.json")
    return {"atomic": atomic, "nc_cell": lattice_matrix(atomic["cell"]),
            "sl_cell": np.asarray(doc["sl_cell_matrix_rows_nm"]), "centers": centers,
            "variant_ids": variant_ids, "rotations": rotations, "laboratory": lab}


TEST_INDICES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1],
                         [1, -2, 3], [2, 1, -1], [3, 0, 1]])


def round_trip_check(config, directory):
    """Compare the reloaded exchange files with the in-memory model.

    Returns maximum absolute differences (nm or rad/nm) for cells, atom
    positions, particle centers, rotations, and laboratory-frame direct and
    reciprocal vectors of TEST_INDICES in every variant.
    """
    scene = build_scene(config)
    loaded = load_bundle(directory)
    ref_nc, ref_sl = np.asarray(scene["ncCell"]), np.asarray(scene["slCell"])
    ref_lab = laboratory_matrix(config, ref_sl)
    ref_sites = sorted((s[0], tuple(np.round(s[1], 8))) for s in _basis_sites(config["atomic"]))
    new_sites = sorted((s[0], tuple(np.round(s[1], 8))) for s in _basis_sites(loaded["atomic"]))
    if [s[0] for s in ref_sites] != [s[0] for s in new_sites]:
        raise ValueError("Round trip changed the atomic species list.")
    report = {
        "nc_cell_nm": float(np.abs(loaded["nc_cell"] - ref_nc).max()),
        "sl_cell_nm": float(np.abs(loaded["sl_cell"] - ref_sl).max()),
        "atom_fractional": float(max(np.abs(np.subtract(a[1], b[1])).max()
                                     for a, b in zip(ref_sites, new_sites))),
        "centers_nm": float(np.abs(loaded["centers"] - [c["xyz"] for c in scene["centers"]]).max()),
        "variant_ids_equal": bool(np.array_equal(loaded["variant_ids"],
                                                 [c["variant"] for c in scene["centers"]])),
        "rotation": float(max(np.abs(r - np.asarray(v["rotation"])).max()
                              for r, v in zip(loaded["rotations"], scene["variants"]))),
        "laboratory": float(np.abs(loaded["laboratory"] - ref_lab).max()),
    }
    direct, recip = 0.0, 0.0
    for r_new, variant in zip(loaded["rotations"], scene["variants"]):
        r_ref = np.asarray(variant["rotation"])
        d_ref = TEST_INDICES @ ref_nc @ r_ref.T @ ref_lab.T  # rows: U R A.T uvw
        d_new = TEST_INDICES @ loaded["nc_cell"] @ r_new.T @ loaded["laboratory"].T
        g_ref = reciprocal_vectors(ref_nc, TEST_INDICES) @ r_ref.T @ ref_lab.T
        g_new = reciprocal_vectors(loaded["nc_cell"], TEST_INDICES) @ r_new.T @ loaded["laboratory"].T
        direct = max(direct, float(np.abs(d_new - d_ref).max()))
        recip = max(recip, float(np.abs(g_new - g_ref).max()))
    report["lab_direct_vectors_nm"] = direct
    report["lab_reciprocal_vectors_nm_inv"] = recip
    report["passed"] = bool(report["variant_ids_equal"] and all(
        v < 1e-7 for k, v in report.items() if isinstance(v, float)))
    return report


# ============================================================================
# symmetry
# ============================================================================
# Separate symmetry operations from physical orientation populations.
#
# Proper lattice rotations permute symmetry-equivalent directions and leave the
# reciprocal-lattice spot set unchanged. Two variants whose relative rotation
# Delta = R_i.T @ R_j is such an operation are the same atomic orientation as
# far as Bragg spot positions go. The particle envelope (or a termination or
# surface marking) can still distinguish them; ``classify_variants`` reports
# each level separately, and ``disorientation_deg`` gives the smallest physical
# misorientation once crystal symmetry is factored out.

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


# ============================================================================
# results
# ============================================================================
# Import external or measured scattering results with their metadata.
#
# A dataset is a CSV file plus a run.json sidecar written by
# ``exchange.run_metadata`` (or by hand). The importer converts q to rad/nm,
# keeps raw intensity and uncertainty, and records backend, version, axes,
# units and normalization. Missing metadata is reported in ``warnings``.
#
# Accepted CSV layouts (header row required; lines starting with # are skipped):
#   profile  q[unit], intensity[, uncertainty]       or 2theta_deg instead of q
#   image    pixel_x, pixel_y, qx, qy[, qz], intensity[, uncertainty]
# Column names are matched case-insensitively. A unit in the header, such as
# ``q_A^-1`` or ``q (1/nm)``, overrides ``units.q`` in run.json.

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


# ============================================================================
# dans
# ============================================================================
# Dans_Diffraction adapter: an independent check of atomic (WAXS) reflections.
#
# Dans_Diffraction is optional (pip install Dans_Diffraction). The adapter only
# calls three long-standing entry points -- Crystal(cif), Scatter.setup_scatter
# and Scatter.intensity -- and generates the hkl list itself, so the comparison
# does not depend on Dans' Cartesian frame. Pin the version you validate with:
# ``run_comparison`` records it. Untested here against a live installation;
# treat the first run as the validation of the adapter itself.
#
# Comparison order follows the integration guide: hkl and q first, then
# reflection absences, then unbroadened |F|^2 ratios. Intensities are compared
# after normalizing each set to its strongest reflection, because the two codes
# use different absolute scales.

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


# ============================================================================
# bornagain
# ============================================================================
# BornAgain bridge: finite assemblies on a detector, including GISAXS geometry.
#
# Three pieces:
#   * ``write_bornagain_script`` turns a configuration plus an ``experiment``
#     block into a standalone BornAgain Python script and a parameter JSON;
#   * ``detector_q`` computes the laboratory q vector of every detector pixel;
#   * ``detector_reference`` evaluates the NanoSuperlattice kinematic (Born)
#     amplitude at exactly those q vectors, giving an equivalent-sampling
#     reference for the "sphere in a homogeneous medium" baseline.
#
# Laboratory frame (BornAgain convention): z is the substrate normal, the
# incident beam travels along +x tilted down by the incidence angle, and y
# completes a right-handed frame. Use the ``laboratory`` block of the
# configuration to place the superlattice in this frame (x_lab = U @ x_SL).
#
# The generated script has NOT been executed by this module. BornAgain renamed
# several classes between releases, so the script resolves names at run time and
# checks its rotation convention. Shape-size conversions, form-factor origins and
# local axes are listed in SHAPES below; confirm them for your BornAgain version
# with the single-particle baseline before trusting a full assembly.
#
# Example ``experiment`` block (lengths nm unless the key says mm):
#   {"geometry": "gisaxs", "wavelength_nm": 0.1, "incidence_deg": 0.2,
#    "detector": {"pixels": [201, 161], "pixel_size_mm": [0.172, 0.172],
#                 "distance_mm": 2000, "beam_center_px": [100, 20]},
#    "media": {"ambient": {"delta": 0, "beta": 0},
#              "particle": {"delta": 6.0e-6, "beta": 1.0e-7},
#              "substrate": {"delta": 4.9e-6, "beta": 7.0e-8}},
#    "born_baseline": false, "particle_gap_nm": 0}

R_E_NM = 2.8179403262e-6  # classical electron radius

# center-to-vertex radius r -> BornAgain constructor arguments. "z_center" is
# the assumed height of the geometric center above the form-factor origin;
# "frame" maps BornAgain local axes to NanoSuperlattice local axes. Entries
# marked verify=True are best understanding, to be confirmed by the baseline.
SHAPES = {
    "sphere": {"names": ["Sphere", "FormFactorFullSphere"],
               "args": lambda r: [r], "z_center": lambda r: r, "verify": False},
    "cube": {"names": ["Box", "FormFactorBox"],
             "args": lambda r: [2*r/np.sqrt(3)]*3, "z_center": lambda r: r/np.sqrt(3), "verify": False},
    "octahedron": {"names": ["PlatonicOctahedron", "FormFactorPlatonicOctahedron"],
                   "args": lambda r: [r*np.sqrt(2)], "z_center": lambda r: r, "verify": True},
    # The Archimedean cuboctahedron is a cube of edge sqrt(2) r with corners
    # cut to the edge midpoints. BornAgain's "Cuboctahedron" is a truncated
    # bipyramid, a different solid, so CantellatedCube is used here.
    "cuboctahedron": {"names": ["CantellatedCube", "FormFactorCantellatedCube"],
                      "args": lambda r: [np.sqrt(2)*r, np.sqrt(2)*r/2],
                      "z_center": lambda r: r/np.sqrt(2), "verify": True},
    "hexagonal_prism": {"names": ["Prism6", "FormFactorPrism6"],
                        "args": lambda r: [r/np.sqrt(1.25), r/np.sqrt(1.25)],
                        "z_center": lambda r: 0.5*r/np.sqrt(1.25), "verify": True},
}


# ---------------------------------------------------------------- geometry
def euler_zxz(matrix):
    """Angles (alpha, beta, gamma) in rad with M = Rz(alpha) @ Rx(beta) @ Rz(gamma)."""
    m = check_rotation(matrix)
    beta = float(np.arccos(np.clip(m[2, 2], -1, 1)))
    if np.sin(beta) > 1e-9:
        return float(np.arctan2(m[0, 2], -m[1, 2])), beta, float(np.arctan2(m[2, 0], m[2, 1]))
    return float(np.arctan2(m[1, 0], m[0, 0])), beta, 0.0


def rotation_zxz(alpha, beta, gamma):
    def rz(t):
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(beta), -np.sin(beta)], [0, np.sin(beta), np.cos(beta)]])
    return rz(alpha) @ rx @ rz(gamma)


def _experiment(config, experiment=None):
    exp = dict(experiment or config.get("experiment") or {})
    if not exp:
        raise ValueError("Add an 'experiment' block (see the bornagain section of integrations.py).")
    det = exp.get("detector", {})
    for key in ("pixels", "pixel_size_mm", "distance_mm", "beam_center_px"):
        if key not in det:
            raise ValueError(f"experiment.detector needs '{key}'.")
    exp.setdefault("geometry", "gisaxs")
    exp.setdefault("wavelength_nm", config.get("scattering", {}).get("wavelength_nm", 0.15406))
    exp.setdefault("incidence_deg", 0.0 if exp["geometry"] == "transmission" else 0.2)
    return exp


def detector_q(experiment):
    """Return q (ny, nx, 3) in rad/nm in the laboratory frame for a flat detector.

    The detector is perpendicular to the direct beam at distance_mm. Pixel
    (i, j) has its center at ((i+0.5-cx)*px, (j+0.5-cy)*py) mm from the
    direct-beam point, measured along lab +y and along the upward normal to
    the beam. q = k_f - k_i, |k| = 2*pi/lambda (elastic).
    """
    det = experiment["detector"]
    nx, ny = det["pixels"]
    px, py = det["pixel_size_mm"]
    cx, cy = det["beam_center_px"]
    alpha = np.radians(experiment.get("incidence_deg", 0.0))
    k = 2 * np.pi / experiment["wavelength_nm"]
    beam = np.array([np.cos(alpha), 0.0, -np.sin(alpha)])
    horizontal = np.array([0.0, 1.0, 0.0])
    vertical = np.cross(beam, horizontal)
    u = (np.arange(nx) + 0.5 - cx) * px
    v = (np.arange(ny) + 0.5 - cy) * py
    points = (det["distance_mm"] * beam + u[None, :, None] * horizontal
              + v[:, None, None] * vertical)
    kf = k * points / np.linalg.norm(points, axis=-1, keepdims=True)
    return kf - k * beam


def electron_density(atomic):
    """Mean electron density (e/nm^3) of the atomic cell from f0(q=0)."""
    volume = abs(np.linalg.det(lattice_matrix(atomic["cell"])))
    total = sum(occ * float(_atomic_factor(sym, np.array([0.0]), {})[0])
                for sym, _, occ, _ in _basis_sites(atomic))
    return total / volume


def refractive_delta(atomic, wavelength_nm):
    """delta = r_e * lambda^2 * rho_e / (2 pi), neglecting anomalous dispersion.

    beta (absorption) requires f'' tables that are not bundled; supply it in
    experiment.media.particle.beta.
    """
    return R_E_NM * wavelength_nm**2 * electron_density(atomic) / (2 * np.pi)


def assembly_particles(config, experiment=None, scene=None):
    """Lab-frame particle list (positions nm, rotations, BornAgain shape arguments)."""
    scene = scene or build_scene(config)
    exp = _experiment(config, experiment)
    lab = laboratory_matrix(config, np.asarray(scene["slCell"]))
    frames = exp.get("bornagain", {}).get("shape_frame", {})
    variants = []
    for i, variant in enumerate(config["variants"]):
        shape = variant.get("shape", config["shape"])
        if shape["type"] not in SHAPES:
            raise ValueError(f"BornAgain has no built-in equivalent of shape {shape['type']!r}; "
                             "approximate it or extend SHAPES.")
        info = SHAPES[shape["type"]]
        r = float(shape["radius_nm"])
        frame = np.asarray(frames.get(shape["type"], np.eye(3)), float)  # BA local -> NC local
        rotation = lab @ np.asarray(scene["variants"][i]["rotation"]) @ check_rotation(frame)
        vertices, _ = particle_mesh(shape)
        variants.append({"shape": shape["type"], "form_factor": info["names"],
                         "args_nm": [float(a) for a in info["args"](r)],
                         "z_center_nm": float(info["z_center"](r)), "verify": info["verify"],
                         "rotation": rotation.tolist(), "euler_zxz_rad": euler_zxz(rotation),
                         "lowest_nm": float((vertices @ (lab @ np.asarray(
                             scene["variants"][i]["rotation"])).T)[:, 2].min())})
    positions = np.array([c["xyz"] for c in scene["centers"]]) @ lab.T
    ids = [c["variant"] for c in scene["centers"]]
    shift = 0.0
    if exp["geometry"] == "gisaxs":
        floor = min(p[2] + variants[v]["lowest_nm"] for p, v in zip(positions, ids))
        shift = float(exp.get("particle_gap_nm", 0.0)) - floor
    positions = positions + [0, 0, shift]
    return {"variants": variants, "z_shift_nm": shift,
            "particles": [{"variant": v, "position_nm": p.tolist()} for p, v in zip(positions, ids)]}


# --------------------------------------------------------------- reference
def detector_reference(config, experiment=None, scene=None):
    """NanoSuperlattice Born-approximation intensity on the detector pixels.

    No refraction, reflection or absorption: this matches a BornAgain run with
    born_baseline=True (substrate equal to the ambient medium).
    """
    scene = scene or build_scene(config)
    exp = _experiment(config, experiment)
    lab = laboratory_matrix(config, np.asarray(scene["slCell"]))
    q_lab = detector_q(exp)
    q_sl = q_lab.reshape(-1, 3) @ lab  # rows: U.T @ q
    settings = config.get("scattering", {}).get("saxs", {})
    contrast = np.asarray(settings.get("contrast", 1), float)
    if contrast.ndim == 0:
        contrast = np.repeat(contrast, len(scene["variants"]))
    intensity = _assembly_intensity(scene, config, q_sl, contrast).reshape(q_lab.shape[:2])
    return {"name": "NanoSuperlattice detector reference", "source": "computed",
            "backend": "NanoSuperlattice", "backend_version": None, "kind": "image",
            "regime": "detector", "image": intensity, "qx": q_lab[..., 0], "qy": q_lab[..., 1],
            "qz": q_lab[..., 2], "uncertainty": None, "wavelength_nm": exp["wavelength_nm"],
            "normalization": "raw Born intensity (contrast^2 nm^6)", "intensity_unit": "model units",
            "axes": {"frame": "laboratory", "convention": "x_lab = U @ x_SL"},
            "geometry": exp, "metadata": {}, "warnings": []}


def compare_images(reference, other, floor=1e-6):
    """Shape comparison of two detector images after scaling each to its maximum.

    Returns the Pearson correlation of log intensities, the median ratio, and
    the max q mismatch between the two pixel grids.
    """
    a, b = np.asarray(reference["image"], float), np.asarray(other["image"], float)
    if a.shape != b.shape:
        raise ValueError(f"Image shapes differ: {a.shape} vs {b.shape}.")
    dq = max(float(np.nanmax(np.abs(np.asarray(reference[k]) - np.asarray(other[k]))))
             for k in ("qx", "qy", "qz"))
    mask = np.isfinite(a) & np.isfinite(b)
    la = np.log10(np.maximum(a[mask] / np.nanmax(a), floor))
    lb = np.log10(np.maximum(b[mask] / np.nanmax(b), floor))
    ratio = (b[mask] / np.nanmax(b)) / np.maximum(a[mask] / np.nanmax(a), floor)
    return {"pixels": int(mask.sum()), "log_correlation": float(np.corrcoef(la, lb)[0, 1]),
            "median_ratio": float(np.median(ratio)), "max_q_mismatch_nm_inv": dq,
            "brightest_pixel_reference": np.unravel_index(np.nanargmax(a), a.shape),
            "brightest_pixel_other": np.unravel_index(np.nanargmax(b), b.shape)}


# ------------------------------------------------------------------ script
def write_bornagain_script(config, directory, experiment=None):
    """Write bornagain_run.py and bornagain_parameters.json into directory."""
    exp = _experiment(config, experiment)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    media = exp.get("media", {})
    if "particle" not in media:
        delta = refractive_delta(config["atomic"], exp["wavelength_nm"])
        media = {**media, "particle": {"delta": delta, "beta": 0.0,
                                       "note": "delta from f0(0); beta not computed"}}
    media.setdefault("ambient", {"delta": 0.0, "beta": 0.0})
    if exp.get("born_baseline") or exp["geometry"] == "transmission":
        media["substrate"] = dict(media["ambient"])
    media.setdefault("substrate", {"delta": 0.0, "beta": 0.0})
    parameters = {"experiment": exp, "media": media, **assembly_particles(config, exp),
                  "q_geometry": {"description": detector_q.__doc__.strip()}}
    parameters["run"] = run_metadata(config, backend="BornAgain", regime="detector",
                                     intensity_unit="BornAgain output units",
                                     normalization="as returned by BornAgain",
                                     axes={"frame": "laboratory: z substrate normal, beam along +x",
                                           "lab_from_sl": "x_lab = U @ x_SL"},
                                     settings={"experiment": exp, "media": media})
    params_path = out / "bornagain_parameters.json"
    params_path.write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    script_path = out / "bornagain_run.py"
    script_path.write_text(SCRIPT, encoding="utf-8")
    return script_path, params_path


SCRIPT = r'''"""BornAgain simulation generated by NanoSuperlattice (integrations.py, SCRIPT string).

Run:  python bornagain_run.py [bornagain_parameters.json]
Writes bornagain_result.csv (pixel_x, pixel_y, qx, qy, qz in nm^-1, intensity)
and bornagain_result.run.json next to the parameter file.

This script was generated, not executed, by NanoSuperlattice. Validate it in
this order: one sphere in a homogeneous medium (born_baseline=true), then a
small periodic array, then refraction and substrate reflection. Compare with
python integrations.py compare-detector.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import bornagain as ba

PARAMS = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("bornagain_parameters.json"))
P = json.loads(PARAMS.read_text(encoding="utf-8"))
EXP = P["experiment"]


def pick(*names):
    for name in names:
        if hasattr(ba, name):
            return getattr(ba, name)
    raise AttributeError("This BornAgain version provides none of: " + ", ".join(names))


Material = pick("RefractiveMaterial", "HomogeneousMaterial")
Compound = pick("Compound", "ParticleComposition")
Simulation = pick("ScatteringSimulation", "GISASSimulation")
R3 = pick("R3", "kvector_t")


def _color():
    """BornAgain >= 25 requires a MaterialColor (display only) after the name."""
    cls = getattr(ba, "MaterialColor")
    for args in ((), (128, 128, 128), (0.5, 0.5, 0.5)):
        try:
            return cls(*args)
        except TypeError:
            continue
    raise RuntimeError("Cannot construct ba.MaterialColor; run python integrations.py probe")


def material(name):
    m = P["media"][name]
    delta, beta = float(m["delta"]), float(m["beta"])
    try:
        return Material(name, delta, beta)
    except TypeError:
        return Material(name, _color(), delta, beta)


def checked_rotation(matrix, angles):
    """RotationEuler(a, b, g) should equal Rz(a) Rx(b) Rz(g). Verify when possible."""
    rot = ba.RotationEuler(*angles)
    target = np.asarray(matrix)
    if hasattr(rot, "transformed"):
        for axis in range(3):
            e = [0.0, 0.0, 0.0]
            e[axis] = 1.0
            v = rot.transformed(R3(*e))
            got = np.array([v.x(), v.y(), v.z()])
            if not np.allclose(got, target[:, axis], atol=1e-6):
                raise RuntimeError("BornAgain RotationEuler convention differs from z-x-z active "
                                   "rotation; adapt checked_rotation() for this version.")
    else:
        print("warning: cannot verify RotationEuler convention in this BornAgain version")
    return rot


def particle(entry):
    v = P["variants"][entry["variant"]]
    ff = pick(*v["form_factor"])(*[a * ba.nm for a in v["args_nm"]])
    core = ba.Particle(material("particle"), ff)
    # Put the geometric center at the compound origin so rotations act about it.
    core.translate(R3(0, 0, -v["z_center_nm"] * ba.nm))
    unit = Compound()
    unit.addComponent(core)
    unit.rotate(checked_rotation(v["rotation"], v["euler_zxz_rad"]))
    unit.translate(R3(*[x * ba.nm for x in entry["position_nm"]]))
    return unit


def sample():
    assembly = Compound()
    for entry in P["particles"]:
        assembly.addComponent(particle(entry))
    layout = ba.ParticleLayout()
    layout.addParticle(assembly)
    top = ba.Layer(material("ambient"))
    top.addLayout(layout)
    multi = ba.MultiLayer()
    multi.addLayer(top)
    multi.addLayer(ba.Layer(material("substrate")))
    return multi


def detector():
    d = EXP["detector"]
    nx, ny = d["pixels"]
    px, py = d["pixel_size_mm"]
    det = ba.RectangularDetector(nx, nx * px, ny, ny * py)
    cx, cy = d["beam_center_px"]
    det.setPerpendicularToDirectBeam(d["distance_mm"], cx * px, cy * py)
    return det


def beam():
    wavelength = EXP["wavelength_nm"] * ba.nm
    alpha = EXP.get("incidence_deg", 0.0) * ba.deg
    try:
        return ba.Beam(1.0, wavelength, alpha)
    except TypeError:
        return ba.Beam(1.0, wavelength, ba.Direction(alpha, 0.0))


def pixel_q():
    d = EXP["detector"]
    nx, ny = d["pixels"]
    px, py = d["pixel_size_mm"]
    cx, cy = d["beam_center_px"]
    a = math.radians(EXP.get("incidence_deg", 0.0))
    k = 2 * math.pi / EXP["wavelength_nm"]
    b = np.array([math.cos(a), 0.0, -math.sin(a)])
    h = np.array([0.0, 1.0, 0.0])
    v = np.cross(b, h)
    u = (np.arange(nx) + 0.5 - cx) * px
    w = (np.arange(ny) + 0.5 - cy) * py
    pts = d["distance_mm"] * b + u[None, :, None] * h + w[:, None, None] * v
    return k * pts / np.linalg.norm(pts, axis=-1, keepdims=True) - k * b


def main():
    sim = Simulation(beam(), sample(), detector())
    result = sim.simulate()
    array = np.asarray(result.array() if hasattr(result, "array") else result.npArray())
    nx, ny = EXP["detector"]["pixels"]
    if array.shape != (ny, nx):
        if array.shape == (nx, ny):
            array = array.T
        else:
            raise RuntimeError(f"Unexpected result shape {array.shape}; expected {(ny, nx)}")
    q = pixel_q()
    out = PARAMS.with_name("bornagain_result.csv")
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pixel_x", "pixel_y", "qx_nm^-1", "qy_nm^-1", "qz_nm^-1", "intensity"])
        for j in range(ny):
            for i in range(nx):
                writer.writerow([i, j, *q[j, i], float(array[j, i])])
    meta = dict(P["run"])
    meta.update(backend_version=getattr(ba, "__version__", "unknown"), name="BornAgain",
                files=[out.name], geometry=EXP)
    out.with_suffix(".run.json").write_text(json.dumps(meta, indent=2))
    print(out)


if __name__ == "__main__":
    main()
'''


# ============================================================================
# atomistic
# ============================================================================
# Finite atomistic assemblies for Debye-equation and real-space codes.
#
# Termination: the fractional point ``origin_frac`` of the atomic lattice
# (default the cell origin, [0,0,0]) is placed at the particle center. Every
# lattice translation of every unique basis site is kept when it lies inside
# the particle envelope, including atoms within ``surface_tol_nm`` of a face.
# Each particle j is then placed at x = R_j @ x_local + t_j.
#
#     from integrations.atomistic import estimate, build_assembly
#     print(estimate(config))            # check counts and memory first
#     asm = build_assembly(config)       # dict of numpy arrays

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


# ============================================================================
# debye
# ============================================================================
# Reference Debye scattering equation for finite atomistic assemblies.
#
#     I(q) = sum_a w_a f_a(q)^2 + 2 sum_{a<b} w_a w_b f_a(q) f_b(q) sinc(q r_ab)
#
# with sinc(x) = sin(x)/x (1 at x = 0), occupancy weights w and the same
# Waasmaier-Kirfel f0 and isotropic B model as waxs.py (no anomalous terms,
# so f is real). Pairs within and between particles are included, so a full
# assembly keeps interparticle interference at wide angles. The result is a
# powder (orientational) average; absolute laboratory orientation is lost but
# pair distances are retained.
#
# This NumPy implementation is the in-repository reference for small systems
# and for checking external codes (AES-Debye, DebUsSy, DebyeCalculator, ...).
# ``bin_width_nm`` switches to a pair-distance histogram, which is much faster
# for >~5000 atoms at the cost of a phase error of about q*bin/2.

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


# ============================================================================
# scattersim
# ============================================================================
# Orientationally averaged SAXS of the finite assembly, for ScatterSim-type comparisons.
#
# ScatterSim and most lattice-model codes report a circularly averaged curve of
# an isotropic powder of superlattice grains. The viewer's SAXS profile is an
# azimuthal mean inside one reciprocal-space slice, which differs whenever the
# assembly is anisotropic. ``saxs_powder_profile`` evaluates the same coherent
# amplitude as saxs.py, averaged over the full sphere of directions.

def fibonacci_hemisphere(n):
    """n nearly uniform unit vectors on the upper hemisphere (Friedel symmetry)."""
    i = np.arange(n) + 0.5
    z = 1 - i / n  # uniform in cos(theta) on [0, 1)
    phi = np.pi * (1 + 5**0.5) * i
    r = np.sqrt(1 - z**2)
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def saxs_powder_profile(config, q, *, n_directions=500, scene=None):
    """Mean of |A(q n)|^2 over n_directions unit vectors n, for each |q|."""
    scene = scene or build_scene(config)
    q = np.asarray(q, float)
    settings = config.get("scattering", {}).get("saxs", {})
    contrast = np.asarray(settings.get("contrast", 1), float)
    if contrast.ndim == 0:
        contrast = np.repeat(contrast, len(scene["variants"]))
    directions = fibonacci_hemisphere(n_directions)
    vectors = (q[:, None, None] * directions[None]).reshape(-1, 3)
    return _assembly_intensity(scene, config, vectors, contrast).reshape(len(q), -1).mean(axis=1)


# ============================================================================
# formats
# ============================================================================
# Writers and job bundles for atomistic and refinement codes.
#
# * XYZ (extended)  -- generic coordinates for Debye calculators (AES-Debye, DebyeCalculator, ...)
# * DISCUS          -- structure files for disorder and diffuse-scattering models
# * AES-Debye       -- XYZ + job description + optional user-supplied command template
# * DebUsSy         -- structure CIF + measured data + mapping of fitted values back
# * ScatterSim      -- neutral parameter file for a superlattice SAXS comparison
#
# Native input formats of AES-Debye, DebUsSy (CLAUDE) and ScatterSim are not
# reproduced here because they must follow the installed version's
# documentation. These bundles hold everything those inputs need, in stated
# units, and ``run_command`` drives any command-line tool from a template.

# -------------------------------------------------------------------- XYZ
def write_xyz(assembly, path, comment="NanoSuperlattice assembly"):
    """Extended XYZ in angstrom with occupancy and particle columns."""
    pos = np.asarray(assembly["positions"]) * 10
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{len(pos)}\n")
        handle.write('Properties=species:S:1:pos:R:3:occupancy:R:1:particle:I:1 '
                     f'units=angstrom comment="{comment}"\n')
        for s, (x, y, z), o, j in zip(assembly["species"], pos, assembly["occupancy"], assembly["particle"]):
            handle.write(f"{s} {x:.6f} {y:.6f} {z:.6f} {o:.4f} {int(j)}\n")
    return Path(path)


def read_xyz(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    n = int(lines[0])
    rows = [line.split() for line in lines[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows]) / 10


# ----------------------------------------------------------------- DISCUS
def _discus_name(symbol):
    return "".join(ch for ch in symbol if ch.isalpha()).upper()[:4]


def discus_cell_text(atomic, title="NanoSuperlattice"):
    """Periodic DISCUS structure: fractional coordinates, B in A^2.

    Layout: title / spcgroup / cell / atoms followed by 'NAME x y z B' rows.
    Check the keyword set against your DISCUS version's structure-file docs.
    """
    from waxs import _basis_sites
    a, b, c, al, be, ga = atomic["cell"]
    lines = [f"title {title}", "spcgroup P1",
             f"cell {a*10:.6f}, {b*10:.6f}, {c*10:.6f}, {al:.4f}, {be:.4f}, {ga:.4f}", "atoms"]
    for symbol, frac, occ, b_iso in _basis_sites(atomic):
        lines.append(f"{_discus_name(symbol):<4} {frac[0]:12.8f} {frac[1]:12.8f} {frac[2]:12.8f} "
                     f"{b_iso*100:10.6f}")
    return "\n".join(lines) + "\n"


def discus_cluster_text(assembly, title="NanoSuperlattice cluster"):
    """Finite cluster as a DISCUS structure in a 1 A orthonormal cell (Cartesian A)."""
    lines = [f"title {title}", "spcgroup P1", "cell 1.0, 1.0, 1.0, 90.0, 90.0, 90.0", "atoms"]
    for s, (x, y, z), b in zip(assembly["species"], np.asarray(assembly["positions"]) * 10,
                               assembly["b_iso_nm2"]):
        lines.append(f"{_discus_name(s):<4} {x:14.6f} {y:14.6f} {z:14.6f} {b*100:10.6f}")
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------- AES-Debye
def write_debye_job(assembly, directory, q, *, config=None, backend="AES-Debye",
                    settings=None, name="assembly"):
    """XYZ coordinates + job.json describing q grid, units and form-factor source."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    xyz = write_xyz(assembly, out / f"{name}.xyz")
    q = np.asarray(q, float)
    job = {"format": "NanoSuperlattice Debye job", "format_version": FORMAT_VERSION,
           "backend": backend, "coordinates": xyz.name, "coordinate_unit": "angstrom",
           "atoms": int(len(assembly["positions"])), "species": sorted(set(assembly["species"])),
           "q_nm_inv": {"min": float(q.min()), "max": float(q.max()), "points": int(len(q))},
           "q_A_inv": {"min": float(q.min() / 10), "max": float(q.max() / 10), "points": int(len(q))},
           "form_factors": "Waasmaier-Kirfel f0 (f0_WaasKirf.dat), no anomalous terms",
           "occupancy": "column 5 of the XYZ file; self term uses w, cross terms w_a*w_b",
           "b_iso_nm2_unique": sorted(set(np.round(assembly["b_iso_nm2"], 8).tolist())),
           "checks": assembly.get("checks", {}), "settings": settings or {}}
    (out / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    np.savetxt(out / "q_grid_nm-1.txt", q, header="q (rad/nm)")
    if config is not None:
        meta = run_metadata(config, backend=backend, regime="waxs",
                            normalization="as returned by the backend", settings=job)
        (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def run_command(template, **paths):
    """Run an external tool from a template such as 'aes-debye {xyz} -o {output}'.

    Placeholders are filled with quoted paths; the command runs without a shell.
    """
    args = [part.format(**{k: str(v) for k, v in paths.items()}) for part in shlex.split(template)]
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"{args[0]} failed ({completed.returncode}): {completed.stderr.strip()}")
    return completed.stdout


# ---------------------------------------------------------------- DebUsSy
def write_debussy_bundle(config, directory, measured=None):
    """Inputs for a DebUsSy refinement prepared outside this repository.

    Writes structure.cif (P1, A), particle_model.json (shape, size, assembly),
    and, when given, measured.dat with q (A^-1), I, sigma. Build the cluster
    database with CLAUDE following the DebUsSy manual, then map the fitted
    values back with config_from_fit.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "structure.cif").write_text(cif_text(config["atomic"], config.get("title", "structure")),
                                       encoding="utf-8")
    model = {"shape": config.get("shape"), "variant_shapes": [v.get("shape") for v in config["variants"]],
             "superlattice_cell_nm": config["superlattice"]["cell"],
             "notes": ["radius_nm is the center-to-vertex distance",
                       "translate the size definition to the CLAUDE cluster convention"]}
    (out / "particle_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    if measured is not None:
        q, i = np.asarray(measured["q"]), np.asarray(measured["intensity"])
        sigma = measured.get("uncertainty")
        note = "measured sigma"
        if sigma is None:
            sigma, note = np.sqrt(np.maximum(i, 1)), "sigma = sqrt(I): PLACEHOLDER, no uncertainty supplied"
        np.savetxt(out / "measured.dat", np.column_stack((q / 10, i, np.asarray(sigma, float))),
                   header=f"q (1/A, 4 pi sin(theta)/lambda)  I  sigma ; {note} ; from "
                          f"{measured.get('source', '')}")
    return out


def config_from_fit(config, fit):
    """Return a new configuration with fitted values applied.

    fit keys (all optional):
      cell_nm: [a,b,c,alpha,beta,gamma] or cell_scale: factor on a,b,c
      radius_nm: center-to-vertex radius for config.shape (and variant shapes)
      b_iso_nm2: {species: B} isotropic displacement per species
      occupancy: {species: value}
      superlattice_scale: factor on the superlattice cell lengths
      fit_info: free-form provenance (residuals, uncertainties, settings)
    """
    new = copy.deepcopy(config)
    if "cell_nm" in fit:
        new["atomic"]["cell"] = [float(v) for v in fit["cell_nm"]]
        lattice_matrix(new["atomic"]["cell"])
    elif "cell_scale" in fit:
        new["atomic"]["cell"][:3] = [float(v) * fit["cell_scale"] for v in config["atomic"]["cell"][:3]]
    if "superlattice_scale" in fit:
        new["superlattice"]["cell"][:3] = [float(v) * fit["superlattice_scale"]
                                           for v in config["superlattice"]["cell"][:3]]
    if "radius_nm" in fit:
        for holder in [new] + new["variants"]:
            if "shape" in holder and "radius_nm" in holder["shape"]:
                holder["shape"]["radius_nm"] = float(fit["radius_nm"])
    for key in ("b_iso_nm2", "occupancy"):
        for site in new["atomic"]["basis"]:
            if site["element"] in fit.get(key, {}):
                site[key] = float(fit[key][site["element"]])
    if "fit_info" in fit:
        new.setdefault("provenance", []).append(fit["fit_info"])
    return new


# -------------------------------------------------------------- ScatterSim
def write_scattersim_bundle(config, directory, q, *, n_directions=500):
    """Parameters for a ScatterSim lattice model plus our matching references.

    Writes scattersim_parameters.json and two NanoSuperlattice profiles:
    the 3D orientational (powder) mean, which matches a circular average of
    an isotropic powder, and the in-slice azimuthal mean used by the viewer.
    """
    from scattering import _slice_frame
    from saxs import simulate_saxs
    from visualize_superlattice import build_scene

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    scene = build_scene(config)
    params = {"format": "NanoSuperlattice ScatterSim parameters", "format_version": FORMAT_VERSION,
              "units": {"length": "nm", "q": "rad/nm"},
              "lattice": {"cell": config["superlattice"]["cell"],
                          "basis_frac": config["superlattice"]["basis"],
                          "repeat": config["superlattice"].get("repeat", [2, 2, 2])},
              "particle": config["shape"],
              "variants": [{"name": v["name"], "rotation": v["rotation"]} for v in scene["variants"]],
              "variant_pattern": config.get("variant_pattern", [0]),
              "contrast": config.get("scattering", {}).get("saxs", {}).get("contrast", 1),
              "averaging": {"powder": f"Fibonacci sphere, {n_directions} directions (Friedel half)",
                            "slice": "uniform azimuth in the configured central slice"},
              "disorder": "none (ideal finite assembly)"}
    (out / "scattersim_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    q = np.asarray(q, float)
    powder = saxs_powder_profile(config, q, n_directions=n_directions, scene=scene)
    write_profile_csv(out / "nanosuperlattice_powder_saxs.csv", q, powder)
    view = config.get("scattering", {}).get("view", {})
    x, y, *_ = _slice_frame(np.asarray(scene["slCell"]), view)
    # A one-pixel map at q = 0; only the in-slice azimuthal profile is needed.
    _, ring = simulate_saxs(scene, config, np.array([0.0]), x, y, q,
                            config.get("scattering", {}).get("saxs", {}))
    write_profile_csv(out / "nanosuperlattice_slice_saxs.csv", q, ring)
    for name, averaging in (("powder", params["averaging"]["powder"]), ("slice", params["averaging"]["slice"])):
        meta = run_metadata(config, regime="saxs", extra={"name": f"NanoSuperlattice {name} SAXS",
                                                          "averaging": averaging})
        (out / f"nanosuperlattice_{name}_saxs.run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


# ============================================================================
# probe
# ============================================================================
# Print the installed BornAgain and Dans_Diffraction APIs the adapters rely on.
#
#     python integrations.py probe > probe.txt
#
# SWIG-wrapped BornAgain classes list their C++ constructor prototypes when
# called with an impossible argument, which is what the adapter needs to know.

class _Impossible:
    pass


def _prototypes(obj):
    try:
        obj(_Impossible())
    except TypeError as exc:
        return str(exc).strip()
    except Exception as exc:  # noqa: BLE001 - report anything the probe hits
        return f"{type(exc).__name__}: {exc}"
    return "accepted an arbitrary object (no prototype list)"


def _methods(cls):
    return ", ".join(m for m in dir(cls) if not m.startswith("_"))


def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def probe_bornagain():
    try:
        import bornagain as ba
    except ImportError as exc:
        print("BornAgain not importable:", exc)
        return
    print("BornAgain", getattr(ba, "__version__", None) or _version("bornagain"))
    public = sorted(n for n in dir(ba) if not n.startswith("_"))
    print("\nALL top-level names:\n" + ", ".join(public))

    # Classes we need to locate after renames in BornAgain 25.
    pattern = re.compile(r"Layout|Detector|Color|Colour|Datafield|Interference|Layer$|Material", re.I)
    for name in public:
        obj = getattr(ba, name)
        if pattern.search(name) and callable(obj):
            print(f"\n=== {name}")
            print("constructors:", _prototypes(obj))
            if inspect.isclass(obj):
                print("methods:", _methods(obj))

    print("\n=== MaterialColor in submodules")
    found = False
    for info in pkgutil.walk_packages(ba.__path__, "bornagain."):
        try:
            module = importlib.import_module(info.name)
        except Exception:  # noqa: BLE001
            continue
        for attr in dir(module):
            if "color" in attr.lower() or "colour" in attr.lower():
                obj = getattr(module, attr)
                found = True
                print(f"{info.name}.{attr}:", _prototypes(obj) if callable(obj) else repr(obj))
    if not found:
        print("none found")

    for name in ("Particle", "Compound", "RotationEuler", "Sample", "Layer"):
        cls = getattr(ba, name, None)
        if cls is not None:
            print(f"\n{name} methods:", _methods(cls))


def probe_dans():
    try:
        import Dans_Diffraction as dif
    except ImportError as exc:
        print("Dans_Diffraction not importable:", exc)
        return
    print("\n\nDans_Diffraction", getattr(dif, "__version__", None) or _version("Dans_Diffraction"))
    try:
        scatter = type(dif.Crystal().Scatter)
    except Exception as exc:  # noqa: BLE001
        print("could not create an empty Crystal:", exc)
        return
    for name in ("setup_scatter", "intensity"):
        print(f"{scatter.__name__}.{name}{inspect.signature(getattr(scatter, name))}")



# ============================================================================
# cli
# ============================================================================
# Command-line entry for the integration adapters. Run from the repository root:
#
#     python integrations.py export examples/silicon_fcc.json exchange/si
#     python integrations.py roundtrip examples/hexagonal_zno.json exchange/zno
#     python integrations.py dans examples/silicon_fcc.json --cif exchange/si/structure.cif
#     python integrations.py bornagain examples/silicon_fcc.json ba/si --experiment exp.json
#     python integrations.py compare-detector examples/silicon_fcc.json ba/si/bornagain_result.csv --experiment exp.json
#     python integrations.py atomistic examples/silicon_fcc.json atoms/si --repeat 1 1 1
#     python integrations.py debye examples/silicon_fcc.json debye/si.csv --repeat 1 1 1 --radius 2
#     python integrations.py aes-debye examples/silicon_fcc.json aes/si --repeat 1 1 1 --radius 2
#     python integrations.py debussy examples/silicon_fcc.json debussy/si --measured data.csv
#     python integrations.py apply-fit examples/silicon_fcc.json fit.json refined.json
#     python integrations.py scattersim examples/silicon_fcc.json scattersim/si
#     python integrations.py symmetry examples/silicon_fcc.json
#     python integrations.py compare-profiles a.csv b.csv

_CLI_HELP = """Command-line entry for the integration adapters. Run from the repository root:

    python integrations.py export examples/silicon_fcc.json exchange/si
    python integrations.py roundtrip examples/hexagonal_zno.json exchange/zno
    python integrations.py dans examples/silicon_fcc.json --cif exchange/si/structure.cif
    python integrations.py bornagain examples/silicon_fcc.json ba/si --experiment exp.json
    python integrations.py compare-detector examples/silicon_fcc.json ba/si/bornagain_result.csv --experiment exp.json
    python integrations.py atomistic examples/silicon_fcc.json atoms/si --repeat 1 1 1
    python integrations.py debye examples/silicon_fcc.json debye/si.csv --repeat 1 1 1 --radius 2
    python integrations.py aes-debye examples/silicon_fcc.json aes/si --repeat 1 1 1 --radius 2
    python integrations.py debussy examples/silicon_fcc.json debussy/si --measured data.csv
    python integrations.py apply-fit examples/silicon_fcc.json fit.json refined.json
    python integrations.py scattersim examples/silicon_fcc.json scattersim/si
    python integrations.py symmetry examples/silicon_fcc.json
    python integrations.py compare-profiles a.csv b.csv
    python integrations.py probe > probe.txt
"""

def _config(path, args=None):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if args is not None:
        if getattr(args, "repeat", None):
            config["superlattice"]["repeat"] = list(args.repeat)
        if getattr(args, "radius", None):
            for holder in [config] + config["variants"]:
                if "shape" in holder:
                    holder["shape"]["radius_nm"] = args.radius
        if getattr(args, "experiment", None):
            config["experiment"] = json.loads(Path(args.experiment).read_text(encoding="utf-8-sig"))
    return config


def _print(obj):
    print(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def cmd_export(args):
    paths = export_bundle(_config(args.config), args.out, scattering=args.scattering)
    for key, path in paths.items():
        print(f"{key:>20}: {path}")


def cmd_roundtrip(args):
    config = _config(args.config)
    export_bundle(config, args.out, scattering=False)
    report = round_trip_check(config, args.out)
    _print(report)
    return 0 if report["passed"] else 1


def cmd_dans(args):
    config = _config(args.config)
    cif = args.cif
    if cif is None:
        cif = Path(args.workdir) / "structure.cif"
        cif.parent.mkdir(parents=True, exist_ok=True)
        cif.write_text(cif_text(config["atomic"]), encoding="utf-8")
    report = run_comparison(config, cif, args.qmax)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    report.pop("table")
    _print(report)
    return 0 if report["passed"] else 1


def cmd_bornagain(args):
    script, params = write_bornagain_script(_config(args.config, args), args.out)
    print(script)
    print(params)


def cmd_detector_reference(args):
    ref = detector_reference(_config(args.config, args))
    _write_image_csv(ref, args.out)
    print(args.out)


def _write_image_csv(dataset, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pixel_x", "pixel_y", "qx_nm^-1", "qy_nm^-1", "qz_nm^-1", "intensity"])
        ny, nx = dataset["image"].shape
        for j in range(ny):
            for i in range(nx):
                writer.writerow([i, j, dataset["qx"][j, i], dataset["qy"][j, i], dataset["qz"][j, i],
                                 dataset["image"][j, i]])


def cmd_compare_detector(args):
    config = _config(args.config, args)
    other = read_dataset(args.result, args.run, regime="detector")
    _print(compare_images(detector_reference(config), other))


def cmd_atomistic(args):
    config = _config(args.config, args)
    print("estimate:", json.dumps(estimate(config)))
    if args.estimate_only:
        return 0
    disorder = {"position_sigma_nm": args.position_sigma, "rotation_sigma_deg": args.rotation_sigma,
                "seed": args.seed}
    asm = build_assembly(config, disorder=disorder, max_atoms=args.max_atoms)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_xyz(asm, out / "assembly.xyz", config.get("title", ""))
    (out / "assembly.discus").write_text(discus_cluster_text(asm, config.get("title", "")), encoding="utf-8")
    (out / "checks.json").write_text(json.dumps({**asm["checks"], "disorder": disorder}, indent=2),
                                     encoding="utf-8")
    _print(asm["checks"])
    return 1 if asm["checks"]["overlap"] or asm["checks"]["duplicate_atoms"] else 0


def cmd_debye(args):
    config = _config(args.config, args)
    q = np.linspace(args.qmin, args.qmax, args.points)
    disorder = {"position_sigma_nm": args.position_sigma, "rotation_sigma_deg": args.rotation_sigma,
                "seed": args.seed}
    asm = build_assembly(config, disorder=disorder, max_atoms=args.max_atoms)
    kwargs = {"bin_width_nm": args.bin}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_profile_csv(out, q, assembly_intensity(asm, q, **kwargs))
    settings = {"atoms": asm["checks"]["atoms"], "bin_width_nm": args.bin, "disorder": disorder}
    meta = run_metadata(config, regime="waxs", intensity_unit="electrons^2",
                        settings=settings, extra={"name": "NanoSuperlattice Debye", "backend_detail": "Debye equation (NumPy reference)"})
    out.with_suffix(".run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(out)
    if args.isolated:
        iso = out.with_name(out.stem + "_isolated.csv")
        write_profile_csv(iso, q, isolated_sum(asm, q, **kwargs))
        meta["name"] = "Debye, sum of isolated particles"
        iso.with_suffix(".run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(iso)


def cmd_aes_debye(args):
    """Write an AES-Debye job; optionally run it and compare with the NumPy reference."""
    config = _config(args.config, args)
    q = np.linspace(args.qmin, args.qmax, args.points)
    asm = build_assembly(config, max_atoms=args.max_atoms)
    out = write_debye_job(asm, args.out, q, config=config,
                          settings={"command": args.command, "bin_width_nm": args.bin})
    reference = out / "nanosuperlattice_debye.csv"
    write_profile_csv(reference, q, assembly_intensity(asm, q, bin_width_nm=args.bin))
    print(out)
    if not args.command:
        print("Job written. Run AES-Debye on assembly.xyz, save q,intensity as CSV, then use "
              "compare-profiles, or pass --command with {xyz} {job} {qgrid} {output} placeholders.")
        return 0
    output = out / "aes_debye_output.csv"
    print(run_command(args.command, xyz=out / "assembly.xyz", job=out / "job.json",
                      qgrid=out / "q_grid_nm-1.txt", output=output))
    _print(compare_profiles(read_dataset(reference, out / "run.json", regime="waxs"),
                            read_dataset(output, out / "run.json", regime="waxs")))


def cmd_debussy(args):
    measured = read_dataset(args.measured) if args.measured else None
    print(write_debussy_bundle(_config(args.config), args.out, measured))


def cmd_apply_fit(args):
    new = config_from_fit(_config(args.config), json.loads(Path(args.fit).read_text(encoding="utf-8")))
    Path(args.out).write_text(json.dumps(new, indent=2), encoding="utf-8")
    print(args.out)


def cmd_scattersim(args):
    q = np.linspace(args.qmin, args.qmax, args.points)
    print(write_scattersim_bundle(_config(args.config, args), args.out, q, n_directions=args.directions))


def cmd_symmetry(args):
    _print(classify_variants(_config(args.config)))


def cmd_probe(args):
    probe_bornagain()
    probe_dans()


def cmd_compare_profiles(args):
    _print(compare_profiles(read_dataset(args.a), read_dataset(args.b)))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python integrations.py", description=_CLI_HELP,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help_text, *, out=True):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config", help="NanoSuperlattice JSON configuration")
        if out:
            p.add_argument("out")
        p.set_defaults(func=func)
        return p

    def geometry_overrides(p):
        p.add_argument("--repeat", type=int, nargs=3, metavar=("NA", "NB", "NC"))
        p.add_argument("--radius", type=float, help="override every radius_nm")

    def disorder(p):
        p.add_argument("--position-sigma", type=float, default=0.0, help="nm")
        p.add_argument("--rotation-sigma", type=float, default=0.0, help="degrees")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--max-atoms", type=int, default=2_000_000)

    p = add("export", cmd_export, "write structure.cif, particles.csv, orientations.json, run.json")
    p.add_argument("--scattering", action=argparse.BooleanOptionalAction, default=None)
    add("roundtrip", cmd_roundtrip, "export then reload and compare geometry")
    p = add("dans", cmd_dans, "compare reflections with Dans_Diffraction", out=False)
    p.add_argument("--cif")
    p.add_argument("--workdir", default="exchange/dans")
    p.add_argument("--qmax", type=float, default=40.0, help="rad/nm")
    p.add_argument("--json", help="write the full reflection table here")
    for name, func, text in (("bornagain", cmd_bornagain, "write a BornAgain script and parameters"),
                             ("detector-reference", cmd_detector_reference,
                              "NanoSuperlattice Born intensity on the detector pixels (CSV)")):
        p = add(name, func, text)
        p.add_argument("--experiment", help="JSON experiment block (overrides config)")
        geometry_overrides(p)
    p = add("compare-detector", cmd_compare_detector, "compare an imported detector CSV", out=False)
    p.add_argument("result")
    p.add_argument("--run")
    p.add_argument("--experiment")
    geometry_overrides(p)
    p = add("atomistic", cmd_atomistic, "build a finite atomistic assembly (XYZ, DISCUS)")
    geometry_overrides(p)
    disorder(p)
    p.add_argument("--estimate-only", action="store_true")
    p = add("debye", cmd_debye, "Debye powder intensity of the atomistic assembly")
    geometry_overrides(p)
    disorder(p)
    p.add_argument("--qmin", type=float, default=5.0)
    p.add_argument("--qmax", type=float, default=50.0)
    p.add_argument("--points", type=int, default=451)
    p.add_argument("--bin", type=float, default=None, help="pair histogram bin (nm); exact if omitted")
    p.add_argument("--isolated", action="store_true", help="also write the sum of isolated particles")
    p = add("aes-debye", cmd_aes_debye, "AES-Debye job bundle (+ optional run and comparison)")
    geometry_overrides(p)
    p.add_argument("--max-atoms", type=int, default=2_000_000)
    p.add_argument("--qmin", type=float, default=5.0)
    p.add_argument("--qmax", type=float, default=50.0)
    p.add_argument("--points", type=int, default=451)
    p.add_argument("--bin", type=float, default=None)
    p.add_argument("--command", help="e.g. \"aes-debye {xyz} --q {qgrid} -o {output}\" "
                                     "(check the installed tool's documented options)")
    p = add("debussy", cmd_debussy, "structure, particle model and measured data for DebUsSy")
    p.add_argument("--measured")
    p = add("apply-fit", cmd_apply_fit, "apply fitted values to a configuration", out=False)
    p.add_argument("fit")
    p.add_argument("out")
    p = add("scattersim", cmd_scattersim, "ScatterSim parameters + matching NanoSuperlattice references")
    geometry_overrides(p)
    p.add_argument("--qmin", type=float, default=0.05)
    p.add_argument("--qmax", type=float, default=2.0)
    p.add_argument("--points", type=int, default=300)
    p.add_argument("--directions", type=int, default=500)
    add("symmetry", cmd_symmetry, "classify variant pairs by lattice, crystal and envelope symmetry", out=False)
    p = sub.add_parser("compare-profiles", help="match peaks and shapes of two profile CSVs")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_compare_profiles)
    sub.add_parser("probe", help="print the installed BornAgain and Dans_Diffraction APIs"
                   ).set_defaults(func=cmd_probe)

    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except (ValueError, KeyError, OSError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
