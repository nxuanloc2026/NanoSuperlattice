"""Exchange files: structure.cif, particles.csv, orientations.json, run.json.

    from integrations.exchange import export_bundle, round_trip_check
    export_bundle(config, "exchange/silicon")
    print(round_trip_check(config, "exchange/silicon"))

CIF files are written in angstrom with a symmetry-expanded P1 basis. All other
files keep nm and rad/nm. The CIF reader also expands simple space-group
operators (x,y,z strings) so structures from databases can be imported.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from crystal_geometry import lattice_matrix
from visualize_superlattice import build_scene
from waxs import _basis_sites

from . import FORMAT_VERSION
from .conventions import check_rotation, energy_kev, laboratory_matrix, reciprocal_vectors

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
        from .conventions import two_theta_deg
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
