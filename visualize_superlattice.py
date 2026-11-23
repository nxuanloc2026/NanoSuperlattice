"""Build a portable, interactive HTML view of an oriented nanocrystal superlattice.

Lengths are in nanometers, cell angles in degrees, and site positions fractional.
Run: python visualize_superlattice.py examples/silicon_fcc.json -o silicon.html
"""

import argparse
from itertools import product
import json
from pathlib import Path
import re

import numpy as np

from crystal_geometry import index_vector, lattice_matrix, orientation_matrix, particle_mesh


HERE = Path(__file__).resolve().parent
PALETTE = ("#2F7794", "#B15A3B", "#6F7E44", "#835DA1", "#A17B31", "#397F72")
ELEMENT_COLORS = {"Si": "#477F9A", "Zn": "#6879A9", "O": "#C65348",
                  "C": "#50595D", "N": "#5669AC", "S": "#AF851D"}


def hex_color(value):
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ValueError(f"Use a six-digit hex color, for example #457BA0. Got {value!r}.")
    return value


def fractionals(values, label):
    """Wrap a periodic basis into one half-open cell, including negative inputs."""
    try:
        data = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain fractional coordinate triples.") from exc
    if data.ndim != 2 or data.shape[1] != 3 or not len(data) or not np.isfinite(data).all():
        raise ValueError(f"{label} must contain one or more finite fractional triples.")
    data = np.mod(data, 1.0)
    data[(np.abs(data) < 1e-10) | (np.abs(data - 1.0) < 1e-10)] = 0.0
    return data


def atomic_detail(atomic, cell):
    entries = atomic["basis"]
    positions = fractionals([entry["frac"] for entry in entries], "atomic.basis")
    atoms, colors, used = [], {}, set()
    for entry, frac in zip(entries, positions):
        symbol = entry["element"]
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Each atomic site needs an element or species label.")
        fallback = ELEMENT_COLORS.get(symbol, PALETTE[len(colors) % len(PALETTE)])
        color = hex_color(entry.get("color", colors.get(symbol, fallback)))
        if symbol in colors and colors[symbol].lower() != color.lower():
            raise ValueError(f"Use one color for all {symbol} sites.")
        colors[symbol] = color
        key = (symbol, tuple(frac.round(10)))
        if key in used:
            continue
        used.add(key)
        # Show boundary sites on opposite faces so the displayed cell looks complete.
        # These periodic images are for drawing, not additional atoms in the basis.
        for corner in product(*[(0, 1) if v == 0 else (0,) for v in frac]):
            atoms.append({"xyz": ((frac + corner) @ cell).tolist(),
                          "element": symbol, "color": color})
    return atoms, colors


def index_label(spec):
    key, values = next(iter(spec.items()))
    brackets = ("[", "]") if key == "direction" else ("(", ")")
    return brackets[0] + " ".join(f"{v:g}" for v in values) + brackets[1]


def build_scene(config):
    """Return a JSON-serializable scene. All plotted coordinates are Cartesian nm."""
    atomic, sl = config["atomic"], config["superlattice"]
    nc_cell, sl_cell = lattice_matrix(atomic["cell"]), lattice_matrix(sl["cell"])
    atoms, colors = atomic_detail(atomic, nc_cell)
    definitions = config["variants"]
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("variants must contain at least one orientation.")
    variants = []
    for i, variant in enumerate(definitions):
        orient = {key: variant[key] for key in ("align", "matrix") if key in variant}
        try:
            rotation = orientation_matrix(orient, nc_cell, sl_cell)
            shape = variant["shape"] if "shape" in variant else config["shape"]
            vertices, faces = particle_mesh(shape)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Variant {i} ({variant.get('name', i)}): {exc}") from exc
        pairs = [{"nc": index_label(pair["nc"]), "sl": index_label(pair["sl"]),
                  "ncVector": index_vector(pair["nc"], nc_cell).tolist(),
                  "slVector": index_vector(pair["sl"], sl_cell).tolist()}
                 for pair in orient.get("align", [])]
        variants.append({"name": variant.get("name", f"Variant {i+1}"),
                         "color": hex_color(variant.get("color", PALETTE[i % len(PALETTE)])),
                         "rotation": rotation.tolist(), "alignments": pairs, "atoms": atoms,
                         "mesh": {"vertices": np.asarray(vertices).tolist(),
                                  "faces": [[int(n) for n in face] for face in faces]}})

    basis = fractionals(sl["basis"], "superlattice.basis")
    if len(np.unique(np.round(basis, 10), axis=0)) != len(basis):
        raise ValueError("superlattice.basis contains periodically equivalent particle centers.")
    repeat = sl.get("repeat", [2, 2, 2])
    if len(repeat) != 3 or any(type(n) is not int or n <= 0 for n in repeat):
        raise ValueError("repeat must be three positive integers.")
    # A pattern of length len(basis) specifies the same motif in every cell.
    # A longer pattern can specify orientation order across cell boundaries.
    pattern = config.get("variant_pattern", [0])
    if not isinstance(pattern, list) or not pattern or any(
        type(i) is not int or not 0 <= i < len(variants) for i in pattern
    ):
        raise ValueError(f"variant_pattern must be a list of indices from 0 to {len(variants)-1}.")
    centers = []
    for cell_id in product(*(range(n) for n in repeat)):
        for site_id, frac in enumerate(basis):
            centers.append({"xyz": ((np.asarray(cell_id) + frac) @ sl_cell).tolist(),
                            "variant": pattern[len(centers) % len(pattern)],
                            "cell": list(cell_id), "site": site_id})
    return {"title": config.get("title", "Nanocrystal superlattice"),
            "description": config.get("description", ""),
            "slCell": sl_cell.tolist(), "ncCell": nc_cell.tolist(), "centers": centers,
            "variants": variants, "elementColors": colors}


def write_viewer(config, output, *, scattering=None, overlays=()):
    """Write a self-contained viewer that can be opened locally in a browser.

    overlays are datasets from integrations.results.read_dataset. Profiles are
    drawn over the matching SAXS or WAXS profile; detector images get their
    own panel with their recorded axes.
    """
    scene = build_scene(config)
    if scattering is None:
        scattering = "scattering" in config
    if scattering:
        from scattering import simulate_scattering
        scene["scattering"] = simulate_scattering(config, scene)
    if overlays:
        from integrations.results import viewer_payload
        scene["imported"] = [viewer_payload(d) for d in overlays]
    payload = json.dumps(scene, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    # A label may contain HTML characters. Keep the embedded script element intact.
    payload = payload.replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")
    template = (HERE / "nanocrystal_viewer.html").read_text(encoding="utf-8")
    if template.count("__SCENE_JSON__") != 1:
        raise ValueError("Viewer template must contain exactly one scene placeholder.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__SCENE_JSON__", payload), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON material and assembly definition")
    parser.add_argument("-o", "--output", type=Path, help="Output HTML (default: beside the input)")
    parser.add_argument("--scattering", action=argparse.BooleanOptionalAction, default=None,
                        help="Simulate SAXS and WAXS, or skip them with --no-scattering. "
                             "Default: simulate if the JSON contains a scattering block.")
    parser.add_argument("--overlay", action="append", default=[], metavar="CSV[,RUN_JSON]",
                        help="Imported result to show with the model. Repeat for several. "
                             "A sibling NAME.run.json or run.json is used when not given.")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        overlays = []
        if args.overlay:
            from integrations.results import parse_overlay_argument
            overlays = [parse_overlay_argument(text) for text in args.overlay]
            for dataset in overlays:
                for warning in dataset["warnings"]:
                    print(f"{dataset['source']}: {warning}")
        output = args.output or args.config.with_suffix(".html")
        if output.suffix.lower() not in {".html", ".htm"}:
            raise ValueError("Choose an output filename ending in .html or .htm.")
        if output.resolve() in {args.config.resolve(), (HERE / "nanocrystal_viewer.html").resolve()}:
            raise ValueError("Choose an output different from the input and viewer template.")
        print(write_viewer(config, output, scattering=args.scattering, overlays=overlays).resolve())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Could not build the viewer: {exc}\n")


if __name__ == "__main__":
    main()
