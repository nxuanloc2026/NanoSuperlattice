"""Writers and job bundles for atomistic and refinement codes.

* XYZ (extended)  -- generic coordinates for Debye calculators (AES-Debye, DebyeCalculator, ...)
* DISCUS          -- structure files for disorder and diffuse-scattering models
* AES-Debye       -- XYZ + job description + optional user-supplied command template
* DebUsSy         -- structure CIF + measured data + mapping of fitted values back
* ScatterSim      -- neutral parameter file for a superlattice SAXS comparison

Native input formats of AES-Debye, DebUsSy (CLAUDE) and ScatterSim are not
reproduced here because they must follow the installed version's
documentation. These bundles hold everything those inputs need, in stated
units, and ``run_command`` drives any command-line tool from a template.
"""

import copy
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np

from crystal_geometry import lattice_matrix

from . import FORMAT_VERSION
from .exchange import cif_text, run_metadata, write_profile_csv


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
    from .scattersim import saxs_powder_profile

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
