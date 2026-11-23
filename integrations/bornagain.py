"""BornAgain bridge: finite assemblies on a detector, including GISAXS geometry.

Three pieces:
  * ``write_bornagain_script`` turns a configuration plus an ``experiment``
    block into a standalone BornAgain Python script and a parameter JSON;
  * ``detector_q`` computes the laboratory q vector of every detector pixel;
  * ``detector_reference`` evaluates the NanoSuperlattice kinematic (Born)
    amplitude at exactly those q vectors, giving an equivalent-sampling
    reference for the "sphere in a homogeneous medium" baseline.

Laboratory frame (BornAgain convention): z is the substrate normal, the
incident beam travels along +x tilted down by the incidence angle, and y
completes a right-handed frame. Use the ``laboratory`` block of the
configuration to place the superlattice in this frame (x_lab = U @ x_SL).

The generated script has NOT been executed by this module. BornAgain renamed
several classes between releases, so the script resolves names at run time and
checks its rotation convention. Shape-size conversions, form-factor origins and
local axes are listed in SHAPES below; confirm them for your BornAgain version
with the single-particle baseline before trusting a full assembly.

Example ``experiment`` block (lengths nm unless the key says mm):
  {"geometry": "gisaxs", "wavelength_nm": 0.1, "incidence_deg": 0.2,
   "detector": {"pixels": [201, 161], "pixel_size_mm": [0.172, 0.172],
                "distance_mm": 2000, "beam_center_px": [100, 20]},
   "media": {"ambient": {"delta": 0, "beta": 0},
             "particle": {"delta": 6.0e-6, "beta": 1.0e-7},
             "substrate": {"delta": 4.9e-6, "beta": 7.0e-8}},
   "born_baseline": false, "particle_gap_nm": 0}
"""

import json
from pathlib import Path

import numpy as np

from crystal_geometry import lattice_matrix, particle_mesh
from saxs import _assembly_intensity
from visualize_superlattice import build_scene
from waxs import _atomic_factor, _basis_sites

from .conventions import check_rotation, laboratory_matrix
from .exchange import run_metadata

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
        raise ValueError("Add an 'experiment' block (see integrations/bornagain.py).")
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


SCRIPT = r'''"""BornAgain simulation generated by NanoSuperlattice (integrations/bornagain.py).

Run:  python bornagain_run.py [bornagain_parameters.json]
Writes bornagain_result.csv (pixel_x, pixel_y, qx, qy, qz in nm^-1, intensity)
and bornagain_result.run.json next to the parameter file.

This script was generated, not executed, by NanoSuperlattice. Validate it in
this order: one sphere in a homogeneous medium (born_baseline=true), then a
small periodic array, then refraction and substrate reflection. Compare with
python -m integrations.cli compare-detector.
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
    raise RuntimeError("Cannot construct ba.MaterialColor; run python -m integrations.probe")


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
