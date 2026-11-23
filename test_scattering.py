"""Numerical regression checks. Run with python -m unittest test_scattering."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from scattering import simulate_scattering
from saxs import simulate_saxs
from visualize_superlattice import build_scene


def example(name):
    path = Path(__file__).parent / "examples" / (name + ".json")
    return json.loads(path.read_text(encoding="utf-8"))


class ScatteringTests(unittest.TestCase):
    def test_sphere_and_fcc_centering(self):
        config = example("silicon_fcc")
        config["shape"] = {"type": "sphere", "radius_nm": 2.0}
        config["superlattice"]["repeat"] = [1, 1, 1]
        a = config["superlattice"]["cell"][0]
        q = np.arange(3) * (2*np.pi/a)
        image, _ = simulate_saxs(build_scene(config), config, q, [1, 0, 0], [0, 1, 0], q, {})
        volume = 4*np.pi*2**3/3
        self.assertAlmostEqual(image[0, 0]/(4*volume)**2, 1.0)
        self.assertLess(image[0, 1]/image[0, 0], 1e-25)
        x = q[2] * 2
        self.assertAlmostEqual(image[0, 2]/image[0, 0], (3*(np.sin(x)-x*np.cos(x))/x**3)**2)

    def test_diamond_structure_factor_and_powder_invariance(self):
        config = example("silicon_fcc")
        config["scattering"]["saxs"].update(pixels=9, profile_points=7, azimuth_samples=4)
        config["scattering"]["waxs"].update(pixels=111, profile_points=101)
        result = simulate_scattering(config)["waxs"]
        q111 = 2*np.pi*np.sqrt(3)/config["atomic"]["cell"][0]
        peaks = result["peaks"]
        self.assertAlmostEqual(peaks[0]["q"], q111)
        self.assertEqual([p["multiplicity"] for p in peaks[:3]], [8, 12, 24])
        np.testing.assert_allclose([p["q"]/q111 for p in peaks[:3]], np.sqrt([3, 8, 11])/np.sqrt(3))
        self.assertFalse(any(np.isclose(p["q"], 2*q111) for p in peaks))  # Si (222)
        second = copy.deepcopy(config)
        second["variant_pattern"] = [0]
        changed = simulate_scattering(second)["waxs"]
        np.testing.assert_allclose(changed["intensity"], result["intensity"])
        self.assertFalse(np.allclose(changed["image"], result["image"]))

    def test_hexagonal_d_spacings(self):
        config = example("hexagonal_zno")
        config["scattering"]["saxs"].update(pixels=7, profile_points=4, azimuth_samples=2)
        config["scattering"]["waxs"].update(pixels=21, profile_points=21)
        result = simulate_scattering(config)
        spacings = [peak["d"] for peak in result["waxs"]["peaks"]]
        a, _, c = config["atomic"]["cell"][:3]
        self.assertTrue(np.any(np.isclose(spacings, a*np.sqrt(3)/2)))  # (100)
        self.assertTrue(np.any(np.isclose(spacings, c/2)))              # (002)
        self.assertFalse(np.any(np.isclose(spacings, c)))               # (001)
        np.testing.assert_allclose(np.cross(result["xCartesian"], result["yCartesian"]),
                                   result["normalCartesian"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
