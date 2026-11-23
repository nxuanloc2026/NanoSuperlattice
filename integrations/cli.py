"""Command-line entry for the integration adapters. Run from the repository root:

    python -m integrations.cli export examples/silicon_fcc.json exchange/si
    python -m integrations.cli roundtrip examples/hexagonal_zno.json exchange/zno
    python -m integrations.cli dans examples/silicon_fcc.json --cif exchange/si/structure.cif
    python -m integrations.cli bornagain examples/silicon_fcc.json ba/si --experiment exp.json
    python -m integrations.cli compare-detector examples/silicon_fcc.json ba/si/bornagain_result.csv --experiment exp.json
    python -m integrations.cli atomistic examples/silicon_fcc.json atoms/si --repeat 1 1 1
    python -m integrations.cli debye examples/silicon_fcc.json debye/si.csv --repeat 1 1 1 --radius 2
    python -m integrations.cli aes-debye examples/silicon_fcc.json aes/si --repeat 1 1 1 --radius 2
    python -m integrations.cli debussy examples/silicon_fcc.json debussy/si --measured data.csv
    python -m integrations.cli apply-fit examples/silicon_fcc.json fit.json refined.json
    python -m integrations.cli scattersim examples/silicon_fcc.json scattersim/si
    python -m integrations.cli symmetry examples/silicon_fcc.json
    python -m integrations.cli compare-profiles a.csv b.csv
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


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
    from .exchange import export_bundle
    paths = export_bundle(_config(args.config), args.out, scattering=args.scattering)
    for key, path in paths.items():
        print(f"{key:>20}: {path}")


def cmd_roundtrip(args):
    from .exchange import export_bundle, round_trip_check
    config = _config(args.config)
    export_bundle(config, args.out, scattering=False)
    report = round_trip_check(config, args.out)
    _print(report)
    return 0 if report["passed"] else 1


def cmd_dans(args):
    from .dans import run_comparison
    from .exchange import cif_text
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
    from .bornagain import write_bornagain_script
    script, params = write_bornagain_script(_config(args.config, args), args.out)
    print(script)
    print(params)


def cmd_detector_reference(args):
    from .bornagain import detector_reference
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
    from .bornagain import compare_images, detector_reference
    from .results import read_dataset
    config = _config(args.config, args)
    other = read_dataset(args.result, args.run, regime="detector")
    _print(compare_images(detector_reference(config), other))


def cmd_atomistic(args):
    from .atomistic import build_assembly, estimate
    from .formats import discus_cluster_text, write_xyz
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
    from .atomistic import build_assembly
    from .debye import assembly_intensity, isolated_sum
    from .exchange import run_metadata, write_profile_csv
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
    from .atomistic import build_assembly
    from .debye import assembly_intensity
    from .exchange import write_profile_csv
    from .formats import run_command, write_debye_job
    from .results import compare_profiles, read_dataset
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
    from .formats import write_debussy_bundle
    from .results import read_dataset
    measured = read_dataset(args.measured) if args.measured else None
    print(write_debussy_bundle(_config(args.config), args.out, measured))


def cmd_apply_fit(args):
    from .formats import config_from_fit
    new = config_from_fit(_config(args.config), json.loads(Path(args.fit).read_text(encoding="utf-8")))
    Path(args.out).write_text(json.dumps(new, indent=2), encoding="utf-8")
    print(args.out)


def cmd_scattersim(args):
    from .formats import write_scattersim_bundle
    q = np.linspace(args.qmin, args.qmax, args.points)
    print(write_scattersim_bundle(_config(args.config, args), args.out, q, n_directions=args.directions))


def cmd_symmetry(args):
    from .symmetry import classify_variants
    _print(classify_variants(_config(args.config)))


def cmd_compare_profiles(args):
    from .results import compare_profiles, read_dataset
    _print(compare_profiles(read_dataset(args.a), read_dataset(args.b)))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m integrations.cli", description=__doc__,
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

    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except (ValueError, KeyError, OSError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
