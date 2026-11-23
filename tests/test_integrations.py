"""Checks for the optional integration adapters. Run: python -m unittest discover tests

Tests that need Dans_Diffraction or BornAgain are skipped when those packages
are not installed; everything else uses NumPy only.
"""

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from nanosuperlattice.crystal_geometry import lattice_matrix
import nanosuperlattice.integrations as nsi
from nanosuperlattice.integrations import laboratory_matrix, reciprocal_vectors, saxs_powder_profile, to_q_nm_inv, two_theta_deg
from nanosuperlattice.visualize_superlattice import build_scene, write_viewer

HERE = Path(__file__).resolve().parent.parent


def example(name):
    return json.loads((HERE / "examples" / f"{name}.json").read_text(encoding="utf-8"))


def triclinic_config():
    """Noncubic cells, a nontrivial matrix rotation and a laboratory frame."""
    c = example("hexagonal_zno")
    c["atomic"] = {"cell": [0.41, 0.53, 0.67, 81.0, 97.5, 104.0],
                   "basis": [{"element": "Zn", "frac": [0.1, 0.2, 0.3], "b_iso_nm2": 0.004},
                             {"element": "O", "frac": [0.6, 0.7, 0.15], "occupancy": 0.8}]}
    c["superlattice"] = {"cell": [11.0, 12.5, 14.0, 85.0, 95.0, 110.0],
                         "basis": [[0, 0, 0], [0.5, 0.4, 0.3]], "repeat": [2, 1, 2]}
    c["shape"] = {"type": "octahedron", "radius_nm": 3.0}
    angle = np.radians(37)
    axis = np.array([1.0, -2.0, 0.5]) / np.linalg.norm([1.0, -2.0, 0.5])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
    c["variants"] = [{"name": "matrix", "matrix": rotation.tolist()},
                     {"name": "second", "matrix": (rotation @ rotation).tolist()}]
    c["variant_pattern"] = [0, 1, 1]
    c["laboratory"] = {"z": {"plane": [0, 0, 1]}, "x": {"direction": [1, 1, 0]}}
    c.pop("scattering")
    return c


class FakeDans:
    """Stands in for Dans_Diffraction, returning the in-repo |F|^2 (or a corrupted variant)."""
    __version__ = "fake"

    def __init__(self, wrong_normals=False):
        self.wrong_normals = wrong_normals
        outer = self

        class Crystal:
            def __init__(self, path):
                atomic = nsi.atomic_from_cif(Path(path).read_text(encoding="utf-8"))
                cell = lattice_matrix(atomic["cell"])

                class Scatter:
                    @staticmethod
                    def setup_scatter(**kwargs):
                        Scatter.kwargs = kwargs

                    @staticmethod
                    def intensity(hkl):
                        return 7.5 * nsi.structure_factor_squared(atomic, hkl)

                class Cell:
                    @staticmethod
                    def Qmag(hkl):
                        # A deliberately wrong reciprocal conversion uses A.T instead of inv(A).
                        m = cell.T if outer.wrong_normals else 2 * np.pi * np.linalg.inv(cell)
                        return np.linalg.norm(np.asarray(hkl) @ m.T, axis=1) / 10

                self.Scatter, self.Cell = Scatter, Cell

        self.Crystal = Crystal


class ExchangeTests(unittest.TestCase):
    def test_round_trip_noncubic_rotation_and_lab_frame(self):
        config = triclinic_config()
        with tempfile.TemporaryDirectory() as tmp:
            nsi.export_bundle(config, tmp)
            report = nsi.round_trip_check(config, tmp)
            self.assertTrue(report["passed"], report)
            orient = json.loads((Path(tmp) / "orientations.json").read_text())
            self.assertIn("x_SL = R @ x_NC", orient["convention"])
            fractions = [v["population_fraction"] for v in orient["variants"]]
            self.assertAlmostEqual(sum(fractions), 1.0)
            run = json.loads((Path(tmp) / "run.json").read_text())
            self.assertEqual(run["units"]["length"], "nm")
            cif = nsi.atomic_from_cif((Path(tmp) / "structure.cif").read_text())
            o = next(s for s in cif["basis"] if s["element"] == "O")
            zn = next(s for s in cif["basis"] if s["element"] == "Zn")
            self.assertAlmostEqual(o["occupancy"], 0.8)
            self.assertAlmostEqual(zn["b_iso_nm2"], 0.004, places=9)
        lab = laboratory_matrix(config, lattice_matrix(config["superlattice"]["cell"]))
        np.testing.assert_allclose(lab @ lab.T, np.eye(3), atol=1e-12)

    def test_cif_space_group_expansion(self):
        ops = ["x,y,z", "x,1/2+y,1/2+z", "1/2+x,y,1/2+z", "1/2+x,1/2+y,z",
               "1/4+x,1/4+y,1/4+z", "1/4+x,3/4+y,3/4+z", "3/4+x,1/4+y,3/4+z", "3/4+x,3/4+y,1/4+z"]
        text = ("data_si\n_cell_length_a 5.431(2)\n_cell_length_b 5.431\n_cell_length_c 5.431\n"
                "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
                "loop_\n_symmetry_equiv_pos_as_xyz\n" + "\n".join(f"'{o}'" for o in ops) +
                "\nloop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
                "Si1 0 0 0\n")
        atomic = nsi.atomic_from_cif(text)
        expected = {tuple(s["frac"]) for s in example("silicon_fcc")["atomic"]["basis"]}
        self.assertEqual({tuple(np.round(s["frac"], 6)) for s in atomic["basis"]}, expected)
        self.assertAlmostEqual(atomic["cell"][0], 0.5431)

    def test_q_conversions(self):
        np.testing.assert_allclose(to_q_nm_inv([0.1], "1/A"), [1.0])
        np.testing.assert_allclose(to_q_nm_inv([1.0], "cycles/nm"), [2 * np.pi])
        q = to_q_nm_inv([28.44], "2theta_deg", 0.15406)
        np.testing.assert_allclose(two_theta_deg(q, 0.15406), [28.44])
        si111 = 2 * np.pi * np.sqrt(3) / 0.5431
        self.assertAlmostEqual(float(two_theta_deg(si111, 0.15406)), 28.44, delta=0.02)


class DansTests(unittest.TestCase):
    def test_silicon_reflections_and_absences(self):
        config = example("silicon_fcc")
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "si.cif"
            cif.write_text(nsi.cif_text(config["atomic"]))
            report = nsi.run_comparison(config, cif, 45.0, module=FakeDans())
        self.assertTrue(report["passed"], report)
        rows = {tuple(r["hkl"]): r for r in report["table"]}
        a = 0.5431
        for hkl, n in (((1, 1, 1), 3), ((2, 2, 0), 8), ((3, 1, 1), 11)):
            self.assertAlmostEqual(rows[hkl]["q_nm_inv"], 2 * np.pi * np.sqrt(n) / a)
            self.assertGreater(rows[hkl]["F2_reference"], 1)
        for absent in ((2, 0, 0), (2, 2, 2), (1, 1, 0)):
            self.assertLess(rows[absent]["F2_reference"], 1e-12)

    def test_zno_exposes_wrong_plane_normals(self):
        config = example("hexagonal_zno")
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "zno.cif"
            cif.write_text(nsi.cif_text(config["atomic"]))
            good = nsi.run_comparison(config, cif, 30.0, module=FakeDans())
            bad = nsi.run_comparison(config, cif, 30.0, module=FakeDans(wrong_normals=True))
        self.assertTrue(good["passed"])
        self.assertFalse(bad["passed"])
        self.assertGreater(bad["max_q_error_nm_inv"], 1)

    def test_oriented_spots_reproduce_waxs_weights_and_powder_invariance(self):
        config = example("silicon_fcc")
        hkl, _ = nsi.hkl_within(lattice_matrix(config["atomic"]["cell"]), 25)
        i = nsi.structure_factor_squared(config["atomic"], hkl)
        spots = nsi.oriented_spots(config, hkl, i)
        self.assertAlmostEqual(sum(s[2].sum() for s in spots), i.sum())
        for _, g, _ in spots:  # rotations preserve |G|, so powder lines are unchanged
            np.testing.assert_allclose(np.linalg.norm(g, axis=1),
                                       np.linalg.norm(reciprocal_vectors(lattice_matrix(config["atomic"]["cell"]), hkl), axis=1))

    @unittest.skipUnless(importlib.util.find_spec("Dans_Diffraction"), "Dans_Diffraction not installed")
    def test_live_dans_diffraction(self):
        config = example("silicon_fcc")
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "si.cif"
            cif.write_text(nsi.cif_text(config["atomic"]))
            report = nsi.run_comparison(config, cif, 45.0)
        table = report.pop("table")
        self.assertTrue(report["absences_agree"], report)
        self.assertLess(report.get("max_q_error_nm_inv", 0), 1e-3)
        # Both codes now use Waasmaier-Kirfel f0 with B = 0, so |F|^2 ratios should agree.
        worst = max(table, key=lambda r: abs(r["F2_external"] / max(r["F2_reference"], 1e-30) - 1)
                    if r["F2_reference"] > 1e-6 * max(t["F2_reference"] for t in table) else 0)
        self.assertLess(report["max_relative_intensity_error"], 0.02, (report, worst))


class ImportTests(unittest.TestCase):
    def test_profile_units_metadata_and_viewer_overlay(self):
        config = example("silicon_fcc")
        config["scattering"]["saxs"].update(pixels=9, profile_points=21, azimuth_samples=4)
        config["scattering"]["waxs"].update(pixels=21, profile_points=51)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "ext.csv").write_text("# external\nq (1/A),I,sigma\n2.0,5,1\n3.2,7,1\n")
            (tmp / "ext.run.json").write_text(json.dumps({"backend": "Test", "regime": "waxs",
                                                          "normalization": "raw"}))
            d = nsi.read_dataset(tmp / "ext.csv")
            np.testing.assert_allclose(d["q"], [20, 32])
            self.assertEqual(d["warnings"], [])
            (tmp / "tth.csv").write_text("2theta_deg,intensity\n28.44,1\n47.3,0.5\n")
            (tmp / "tth.run.json").write_text(json.dumps({"backend": "Test", "regime": "waxs",
                                                          "normalization": "raw", "wavelength_nm": 0.15406}))
            t = nsi.read_dataset(tmp / "tth.csv")
            self.assertAlmostEqual(t["q"][0], 2 * np.pi * np.sqrt(3) / 0.5431, delta=0.02)
            with self.assertRaises(ValueError):  # 2-theta without a wavelength cannot become q
                nsi.read_dataset(tmp / "tth.csv", tmp / "ext.run.json")
            html = write_viewer(config, tmp / "v.html", overlays=[d])
            self.assertIn('"imported"', html.read_text(encoding="utf-8"))

    def test_detector_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "det.csv"
            rows = ["pixel_x,pixel_y,qx_nm^-1,qy_nm^-1,qz_nm^-1,intensity"]
            rows += [f"{i},{j},0,{i*0.1},{j*0.1},{i+j}" for j in range(3) for i in range(4)]
            path.write_text("\n".join(rows))
            d = nsi.read_dataset(path, regime="detector")
        self.assertEqual(d["kind"], "image")
        self.assertEqual(d["image"].shape, (3, 4))
        self.assertAlmostEqual(d["qz"][2, 1], 0.2)

    def test_compare_profiles_matches_peaks(self):
        q = np.linspace(1, 10, 500)
        a = {"q": q, "intensity": np.exp(-(q - 4) ** 2 / 0.02) + 0.5 * np.exp(-(q - 7) ** 2 / 0.02)}
        b = {"q": q, "intensity": 3 * np.exp(-(q - 4.05) ** 2 / 0.02) + 1.4 * np.exp(-(q - 7) ** 2 / 0.02)}
        report = nsi.compare_profiles(a, b)
        self.assertEqual(len(report["matches"]), 2)
        self.assertLess(report["max_peak_shift_nm_inv"], 0.06)


class BornAgainTests(unittest.TestCase):
    def experiment(self):
        return json.loads((HERE / "examples" / "experiment_gisaxs.json").read_text())

    def test_euler_round_trip(self):
        rng = np.random.default_rng(3)
        for _ in range(100):
            m = nsi.rotation_zxz(*rng.uniform([-np.pi, 0, -np.pi], [np.pi, np.pi, np.pi]))
            np.testing.assert_allclose(nsi.rotation_zxz(*nsi.euler_zxz(m)), m, atol=1e-9)
        for m in (np.eye(3), np.diag([1.0, -1, -1])):
            np.testing.assert_allclose(nsi.rotation_zxz(*nsi.euler_zxz(m)), m, atol=1e-9)

    def test_sphere_detector_reference_is_form_factor(self):
        config = example("silicon_fcc")
        config["shape"] = {"type": "sphere", "radius_nm": 5.0}
        config["superlattice"].update(repeat=[1, 1, 1], basis=[[0, 0, 0]])
        ref = nsi.detector_reference(config, self.experiment())
        x = 5 * np.sqrt(ref["qx"] ** 2 + ref["qy"] ** 2 + ref["qz"] ** 2)
        form = (4 / 3 * np.pi * 125 * 3 * (np.sin(x) - x * np.cos(x)) / x ** 3) ** 2
        np.testing.assert_allclose(ref["image"], form, rtol=1e-10)

    def test_detector_q_is_elastic(self):
        exp = self.experiment()
        q = nsi.detector_q(exp)
        k = 2 * np.pi / exp["wavelength_nm"]
        a = np.radians(exp["incidence_deg"])
        kf = q + k * np.array([np.cos(a), 0, -np.sin(a)])
        np.testing.assert_allclose(np.linalg.norm(kf, axis=-1), k)

    def test_script_and_parameters(self):
        config = example("silicon_fcc")
        config["superlattice"]["repeat"] = [1, 1, 1]
        with tempfile.TemporaryDirectory() as tmp:
            script, params = nsi.write_bornagain_script(config, tmp, self.experiment())
            ast.parse(script.read_text())
            p = json.loads(params.read_text())
        self.assertEqual(len(p["particles"]), 4)
        lowest = min(e["position_nm"][2] + p["variants"][e["variant"]]["lowest_nm"] for e in p["particles"])
        self.assertAlmostEqual(lowest, 0.0, places=9)
        self.assertEqual(p["media"]["substrate"], p["media"]["ambient"])  # born_baseline
        v = p["variants"][0]
        np.testing.assert_allclose(nsi.rotation_zxz(*v["euler_zxz_rad"]), v["rotation"], atol=1e-9)

    def test_shape_conversions_preserve_volume(self):
        expected = {"cube": lambda a: a[0] * a[1] * a[2],
                    "octahedron": lambda a: np.sqrt(2) / 3 * a[0] ** 3,
                    "cuboctahedron": lambda a: a[0] ** 3 - 8 * (a[1] ** 3) / 6,
                    "hexagonal_prism": lambda a: 3 * np.sqrt(3) / 2 * a[0] ** 2 * a[1],
                    "sphere": lambda a: 4 / 3 * np.pi * a[0] ** 3}
        for name, volume in expected.items():
            shape = {"type": name, "radius_nm": 2.0}
            args = nsi.SHAPES[name]["args"](2.0)
            self.assertAlmostEqual(volume(args), nsi._shape_volume(shape), places=6, msg=name)

    @unittest.skipUnless(importlib.util.find_spec("bornagain"), "BornAgain not installed")
    def test_live_bornagain_sphere_baseline(self):
        import subprocess
        config = example("silicon_fcc")
        config["shape"] = {"type": "sphere", "radius_nm": 5.0}
        config["superlattice"].update(repeat=[1, 1, 1], basis=[[0, 0, 0]])
        exp = self.experiment()
        with tempfile.TemporaryDirectory() as tmp:
            script, params = nsi.write_bornagain_script(config, tmp, exp)
            subprocess.run([sys.executable, str(script), str(params)], check=True)
            other = nsi.read_dataset(Path(tmp) / "bornagain_result.csv", regime="detector")
        report = nsi.compare_images(nsi.detector_reference(config, exp), other)
        self.assertGreater(report["log_correlation"], 0.99, report)


class AtomisticTests(unittest.TestCase):
    def small(self, radius=1.5, repeat=(1, 1, 1)):
        c = example("silicon_fcc")
        c["superlattice"]["repeat"] = list(repeat)
        c["shape"]["radius_nm"] = radius
        return c

    def test_counts_duplicates_and_overlap(self):
        c = self.small()
        c["shape"] = {"type": "sphere", "radius_nm": 2.0}
        asm = nsi.build_assembly(c)
        est = nsi.estimate(c)["atoms_estimate"]
        self.assertLess(abs(asm["checks"]["atoms"] / est - 1), 0.08)
        self.assertEqual(asm["checks"]["duplicate_atoms"], 0)
        self.assertFalse(asm["checks"]["overlap"])
        c["superlattice"]["cell"] = [5, 5, 5, 90, 90, 90]  # spheres of radius 2 at 3.5 nm spacing
        self.assertTrue(nsi.build_assembly(c)["checks"]["overlap"])

    def test_placement_uses_rotation_and_center(self):
        c = self.small()
        scene = build_scene(c)
        asm = nsi.build_assembly(c, scene)
        local = nsi.particle_atoms(c["atomic"], c["shape"])[0]
        j = 3  # four FCC sites in one cell; site 3 has a nonzero center
        center = scene["centers"][j]
        r = np.asarray(scene["variants"][center["variant"]]["rotation"])
        np.testing.assert_allclose(asm["positions"][asm["particle"] == j], local @ r.T + center["xyz"])

    def test_disorder_is_seeded(self):
        c = self.small()
        d = {"position_sigma_nm": 0.3, "rotation_sigma_deg": 2, "seed": 4}
        a = nsi.build_assembly(c, disorder=d)["positions"]
        b = nsi.build_assembly(c, disorder=d)["positions"]
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.allclose(a, nsi.build_assembly(c)["positions"]))

    def test_exports(self):
        asm = nsi.build_assembly(self.small(1.0))
        with tempfile.TemporaryDirectory() as tmp:
            path = nsi.write_xyz(asm, Path(tmp) / "a.xyz")
            species, pos = nsi.read_xyz(path)
            np.testing.assert_allclose(pos, asm["positions"], atol=1e-6)
            job = nsi.write_debye_job(asm, tmp, np.linspace(5, 50, 10))
            self.assertEqual(json.loads((job / "job.json").read_text())["atoms"], len(species))
        text = nsi.discus_cell_text(example("hexagonal_zno")["atomic"])
        self.assertIn("cell 3.250100, 3.250100, 5.207100, 90.0000, 90.0000, 120.0000", text)
        self.assertEqual(text.count("\nZN "), 2)

    def test_external_command_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "o.txt"
            nsi.run_command(f'"{sys.executable}" -c "import sys; open(sys.argv[1],\'w\').write(\'ok\')" {{output}}',
                                output=out)
            self.assertEqual(out.read_text(), "ok")

    def test_fit_mapping(self):
        c = example("silicon_fcc")
        new = nsi.config_from_fit(c, {"cell_scale": 1.01, "radius_nm": 5.0,
                                          "b_iso_nm2": {"Si": 0.004}, "fit_info": {"chi2": 1.1}})
        self.assertAlmostEqual(new["atomic"]["cell"][0], 0.5431 * 1.01)
        self.assertEqual(new["shape"]["radius_nm"], 5.0)
        self.assertTrue(all(s["b_iso_nm2"] == 0.004 for s in new["atomic"]["basis"]))
        self.assertEqual(c["shape"]["radius_nm"], 4.8)  # original untouched
        build_scene(new)


class DebyeTests(unittest.TestCase):
    def test_two_atoms_and_histogram(self):
        from nanosuperlattice.waxs import _atomic_factor
        q = np.linspace(0.5, 50, 200)
        x = np.array([[0, 0, 0], [0.3, 0.1, 0]])
        f = _atomic_factor("Si", q, {})
        r = np.linalg.norm(x[1])
        np.testing.assert_allclose(nsi.debye_intensity(x, ["Si", "Si"], q),
                                   2 * f ** 2 * (1 + np.sin(q * r) / (q * r)))
        c = example("silicon_fcc")
        c["superlattice"].update(repeat=[1, 1, 1], basis=[[0, 0, 0]])
        c["shape"]["radius_nm"] = 1.2
        asm = nsi.build_assembly(c)
        exact = nsi.assembly_intensity(asm, q)
        hist = nsi.assembly_intensity(asm, q, bin_width_nm=1e-4)
        self.assertLess(np.abs(hist - exact).max() / exact.max(), 1e-4)

    def test_debye_peaks_match_powder_reflections(self):
        c = example("silicon_fcc")
        c["superlattice"].update(repeat=[1, 1, 1], basis=[[0, 0, 0]])
        c["shape"]["radius_nm"] = 2.0
        q = np.linspace(15, 42, 400)
        asm = nsi.build_assembly(c)
        peaks = nsi.profile_peaks(q, nsi.assembly_intensity(asm, q, bin_width_nm=2e-4), 0.2)
        a = 0.5431
        for n in (3, 8, 11):
            target = 2 * np.pi * np.sqrt(n) / a
            self.assertLess(min(abs(p - target) for p in peaks) / target, 0.02, (n, peaks))

    def test_isolated_sum_equals_assembly_for_single_particle(self):
        c = example("silicon_fcc")
        c["superlattice"].update(repeat=[1, 1, 1], basis=[[0, 0, 0]])
        c["shape"]["radius_nm"] = 0.9
        asm = nsi.build_assembly(c)
        q = np.linspace(5, 40, 50)
        np.testing.assert_allclose(nsi.isolated_sum(asm, q), nsi.assembly_intensity(asm, q))


class SymmetryAndPowderTests(unittest.TestCase):
    def test_operation_counts_and_classification(self):
        si, zno = example("silicon_fcc"), example("hexagonal_zno")
        self.assertEqual(len(nsi.crystal_rotations(si["atomic"])), 24)
        self.assertEqual(len(nsi.crystal_rotations(zno["atomic"])), 6)
        verdicts = [p["verdict"] for p in nsi.classify_variants(si)["pairs"]]
        self.assertEqual(set(verdicts), {"physically distinct atomic orientation"})
        r = np.asarray(build_scene(si)["variants"][0]["rotation"])
        four = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        si["variants"] = [si["variants"][0], {"name": "O1 . 4z", "matrix": (r @ four).tolist()}]
        si["variant_pattern"] = [0, 1]
        pair = nsi.classify_variants(si)["pairs"][0]
        self.assertTrue(pair["indistinguishable_particles"])
        self.assertAlmostEqual(pair["disorientation_deg"], 0.0, places=6)
        si["shape"] = {"type": "hexagonal_prism", "radius_nm": 3.0}  # envelope breaks the 4-fold
        self.assertEqual(nsi.classify_variants(si)["pairs"][0]["verdict"],
                         "same atomic orientation; envelope distinguishes")

    def test_powder_saxs_of_sphere(self):
        c = example("silicon_fcc")
        c["shape"] = {"type": "sphere", "radius_nm": 3.0}
        c["superlattice"].update(repeat=[1, 1, 1], basis=[[0, 0, 0]])
        q = np.linspace(0.05, 2, 25)
        x = 3 * q
        form = (4 / 3 * np.pi * 27 * 3 * (np.sin(x) - x * np.cos(x)) / x ** 3) ** 2
        np.testing.assert_allclose(saxs_powder_profile(c, q, n_directions=20), form, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
