# NanoSuperlattice

NanoSuperlattice shows how each nanocrystal's atomic lattice is oriented within a larger particle assembly and predicts the resulting X-ray scattering. Give it the two unit cells, their atomic and particle positions, and an orientation relationship. Python and NumPy generate a standalone HTML file with linked 3D views, scattering maps, and downloadable numerical data.

## Quick start

Install Python 3.9 or later. Clone the repository, install the package (this also installs NumPy), and generate the silicon example.

```sh
git clone https://github.com/nxuanloc2026/NanoSuperlattice.git
cd NanoSuperlattice
python -m pip install -e .
nanosuperlattice examples/silicon_fcc.json -o silicon.html
```

`python -m nanosuperlattice` runs the same command from a checkout without installing anything except NumPy. The test release on TestPyPI installs with:

```sh
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nanosuperlattice
```

Optional extras pull in the external codes used by the integration commands: `pip install "nanosuperlattice[bornagain]"`, `"nanosuperlattice[dans]"` or `"nanosuperlattice[all]"`. The example configurations are in the [examples folder](https://github.com/nxuanloc2026/NanoSuperlattice/tree/main/examples) of the repository.

Open `silicon.html` in a browser. It works offline. Click a particle in the assembly on the left to see its atomic unit cell on the right. The two panels share a camera and use separate length scales.

For silicon, select **O1**, choose the [110] NC ∥ [111] SL alignment, and click **Look along alignment**. Switch to the second alignment while keeping that camera position. Cycle through **O1**, **O2**, and **O3** to see the in-plane arrow turn through three azimuths. **NC** denotes the nanocrystal atomic lattice and **SL** the superlattice of particle centers.

### Included examples

| Input | What to look for |
|---|---|
| [Silicon cuboctahedra on an FCC superlattice](https://github.com/nxuanloc2026/NanoSuperlattice/blob/main/examples/silicon_fcc.json) | Three orientation variants share a [110] atomic zone axis. The first three powder reflections are 111, 220, and 311 at approximately 20.04, 32.72, and 38.37 nm⁻¹. |
| [Wurtzite ZnO on a hexagonal superlattice](https://github.com/nxuanloc2026/NanoSuperlattice/blob/main/examples/hexagonal_zno.json) | Switch between basal and inclined variants to see how either the (001) or the (101) atomic plane can lie parallel to the same assembly layers. |

To build the second example, run `nanosuperlattice examples/hexagonal_zno.json -o zno.html`. The particle-site assignments in both demonstrations are illustrative.

### Reading the scattering output

| View | What it measures in the model | CSV output |
|---|---|---|
| Small angle (SAXS) | Interference between particles, including their envelopes, orientations, and positions. The profile averages over azimuth in the selected reciprocal-space slice. | Map, profile, and local maxima with apparent spacings |
| Wide angle (WAXS) | Diffraction from the atomic basis. The map shows the specified orientation population. The powder profile averages over all directions. | Map, profile, and indexed reflections with spacing, multiplicity, and scattering angle |

To enter your own structure, see [Define another material](#define-another-material). For q ranges, contrast, peak widths, and modeling assumptions, see [Simulate SAXS and WAXS](#simulate-saxs-and-waxs).

## Define another material

Copy an example JSON file and edit the following fields.

| Field | Meaning |
|---|---|
| `atomic.cell` | Six values `[a,b,c,alpha,beta,gamma]` for the atomic lattice. Lengths are in nm and angles in degrees. Alpha is between b and c, beta between a and c, gamma between a and b. Any nondegenerate triclinic cell is supported. |
| `atomic.basis` | All atomic sites in one unit cell as species labels and fractional positions. Supply symmetry-expanded positions when starting from a CIF asymmetric unit. Boundary atoms are drawn on both faces for visibility. |
| `superlattice.cell` | The independent translation cell of the particle centers, in the same six-parameter format. |
| `superlattice.basis` | Fractional particle centers within one superlattice cell. One site gives a primitive motif. Face or body centering can be given explicitly. |
| `superlattice.repeat` | Number of displayed cells along the three direct lattice vectors. |
| `shape` | Particle envelope in the nanocrystal's local Cartesian coordinates. Choose `sphere`, `cube`, `octahedron`, `cuboctahedron`, or `hexagonal_prism` and specify `radius_nm`, the center-to-vertex distance. The regular hexagonal prism has equal edge lengths and its symmetry axis along local z. A variant may override `shape`. |
| `variants` | Each variant has a name, optional hex color and either two signed alignment pairs or one rotation matrix. |
| `variant_pattern` | Zero-based variant indices, cycled over particle sites. Sites follow basis order within each cell. The third cell index advances fastest, then the second, then the first. Use one entry per site to specify an entire displayed arrangement. Default is variant 0 throughout. |

For a measured shape use `"shape": {"type":"custom","vertices_nm":[...],"faces":[...]}`. Vertices are local Cartesian triples relative to the intended particle center. Faces are ordered lists of zero-based vertex indices forming a closed surface. Use triangles or convex planar polygon faces with consistent winding. The crystal axes come from the specified atomic cell, so orient the mesh in that frame before exporting its vertices.

### Two ways to enter an orientation

This example aligns a basal plane and an in-plane direct lattice direction. Use directed three-index notation, including for hexagonal cells.

```json
{
  "name": "Basal plane",
  "align": [
    {"nc": {"plane": [0,0,1]}, "sl": {"plane": [0,0,1]}},
    {"nc": {"direction": [1,0,0]}, "sl": {"direction": [1,0,0]}}
  ]
}
```

The first pair fixes an axis, and the second fixes the rotation about it. The two axes on each side must be nonparallel and subtend the same angle. For lattice matrix A with the direct basis vectors in its **rows**, the code converts a direct [uvw] to Cartesian Aᵀ [u,v,w]ᵀ, and the normal to (hkl) to A⁻¹ [h,k,l]ᵀ. This distinction matters for noncubic cells. Fractional row positions map as f A. Local Cartesian x is parallel to a, y is in the ab plane, and z is parallel to a × b.

Alternatively supply a proper rotation in orthonormal Cartesian coordinates, with the active column-vector convention **x_SL = R x_NC**. For example, a +90° rotation about Cartesian z is

```json
{"name": "Quarter turn", "matrix": [[0,-1,0],[1,0,0],[0,0,1]]}
```

For a particle at superlattice position t, any atomic or shape coordinate x is placed at **t + R x**. Both panels use this same rotation. Particle arrangements and orientation populations are inputs to the model.

## Simulate SAXS and WAXS

Both example files enable scattering. For an existing material without a scattering block, add the command-line flag. The geometry-only mode is also available.

```sh
nanosuperlattice material.json --scattering -o scattering.html
nanosuperlattice material.json --no-scattering -o geometry.html
```

Open the **SAXS** and **WAXS** views in the generated HTML to inspect the maps and profiles. CSV downloads preserve the calculated intensities before display normalization. A map pixel is evaluated at **q = qₓ eₓ + qᵧ eᵧ** in a central slice of reciprocal space. A fixed slice makes it possible to orbit the real-space model without changing the scattering experiment. The maps are ideal reciprocal-space sections. Modeling a particular transmission detector or grazing-incidence experiment requires its Ewald-sphere intersection and experimental geometry.

### Sample orientation and resolution

Insert this object under the top-level key `scattering` to look down [111] of a cubic superlattice, with [−2 1 1] horizontal.

```json
{
  "wavelength_nm": 0.15406,
  "view": {
    "normal": {"direction": [1, 1, 1]},
    "horizontal": {"direction": [-2, 1, 1]}
  },
  "saxs": {"q_max_nm_inv": 1.65, "pixels": 121, "profile_points": 201, "azimuth_samples": 128, "contrast": 1},
  "waxs": {"q_max_nm_inv": 55, "pixels": 241, "profile_points": 551, "sigma_nm_inv": 0.55}
}
```

Both orientation specifications use the **superlattice** direct or reciprocal basis. Supply a `direction` or a `plane` normal under each key. An explicit horizontal axis must lie in the slice. The vertical axis is the slice normal crossed with the horizontal axis. The default normal is the (001) SL plane normal. The map spans ±q_max along each axis, and the profile spans 0 through q_max. Set an odd pixel count to include the origin. Choose pixel and radial spacings finer than a peak width when measuring peak heights. The wavelength converts q to the elastic scattering angle via 2θ = 2 asin(qλ/4π). Reciprocal-space coordinates and intensities are calculated independently of that display conversion.

### Small-angle model

For site j at position tⱼ with orientation Rⱼ, the simulator sums the complex amplitudes **Δρⱼ Fⱼ(Rⱼᵀq) exp(i q·tⱼ)** before squaring the modulus. F is the Fourier transform of a homogeneous particle envelope. Spheres use their analytic form factor, and polyhedra use a surface integral over the mesh. Changing the particle shape, size, orientation or superlattice repeat count changes the calculated pattern. `contrast` can be one real particle-minus-medium scattering density or a list in variant order. With electron-density contrast in e/nm³, raw intensity is in e², before instrumental scaling. At q = 0 the result is (Σ ΔρⱼVⱼ)².

The SAXS profile averages intensities over azimuth **in the chosen slice**; Friedel symmetry permits sampling half a circle. Increase `azimuth_samples` and `profile_points` to check convergence for larger assemblies. Interior local maxima are estimated by quadratic interpolation. Their apparent length 2π/q is useful for comparison. Particle form factors and finite-size fringes can also produce maxima, so a SAXS maximum alone does not identify a lattice plane. [Small-angle scattering theory](https://journals.iucr.org/j/issues/2021/06/00/gj5274/)

### Wide-angle model

For each atomic reciprocal vector **G** with Miller indices (h,k,l), the kinematic intensity uses **|Σⱼ fⱼ(q) exp[2πi(hxⱼ+kyⱼ+lzⱼ)]|²**. The calculation uses unique fractional atomic sites. Periodic boundary images drawn by the viewer do not add to the basis. Each variant rotates G into the SL frame. Intensities add with the fraction of displayed sites in each variant, assuming equal diffracting power per particle and incoherent scattering between particles at wide angles. An isotropic three-dimensional Gaussian of standard deviation `sigma_nm_inv` broadens each reciprocal point. The width is an adjustable effective width in q, independent of the mesh size.

The WAXS profile is a **three-dimensional powder orientational mean** of the broadened reflections. Rotating a variant affects the slice while leaving the powder profile invariant. For each powder line the table lists a representative hkl, the summed multiplicity, q, d = 2π/q, 2θ and the integrated reciprocal-space weight Σ|F|². Distinct families that share the same spacing contribute to the same powder line. Individual atomic sites can specify `occupancy` (default 1) and an isotropic displacement parameter `b_iso_nm2` (default 0) for WAXS amplitudes. The model omits absorption, anomalous dispersion, detector polarization and grazing-incidence refraction.

The bundled `nanosuperlattice/f0_WaasKirf.dat` provides q-dependent X-ray scattering factors for tabulated neutral atoms and ions. For other species labels, configure `waxs.form_factors` as a mapping from label to a constant electron amplitude or to `{"a":[...],"b":[...],"c":...}` for f(q) = Σ aᵢ exp(−bᵢs²) + c, s = q/(40π) in Å⁻¹ when q is in nm⁻¹. [IUCr structure factor](https://dictionary.iucr.org/Structure_factor) · [Waasmaier & Kirfel](https://doi.org/10.1107/S0108767394013292) · [DABAX data](https://github.com/oasys-kit/DabaxFiles/blob/main/f0_WaasKirf.dat)

## Use from Python

```python
import json
from pathlib import Path
from nanosuperlattice import build_scene, simulate_scattering, write_viewer

config = json.loads(Path("examples/hexagonal_zno.json").read_text(encoding="utf-8"))
scene = build_scene(config)
print(scene["variants"][1]["rotation"])
result = simulate_scattering(config, scene)
print(result["waxs"]["peaks"][0])  # q in nm⁻¹, d in nm
print(result["saxs"]["maxima"])    # local maxima of the slice-averaged curve
write_viewer(config, "zno.html", scattering=True)
```

## Connect external software

The optional module `nanosuperlattice.integrations` (command `nanosuperlattice-integrations`) exports the model as CIF, particle-center, orientation and run-metadata files. It imports external results, with their units and provenance, for overlay in the viewer. It also bridges to Dans_Diffraction, CrystalMaker / SingleCrystal, BornAgain, AES-Debye, DebUsSy, ScatterSim and DISCUS, and includes a NumPy Debye-equation reference for finite atomistic assemblies.

```sh
nanosuperlattice-integrations export examples/silicon_fcc.json exchange/si
nanosuperlattice-integrations roundtrip examples/hexagonal_zno.json exchange/zno
nanosuperlattice examples/silicon_fcc.json --overlay exchange/si/scattering_waxs.csv
```

[INTEGRATIONS.md](https://github.com/nxuanloc2026/NanoSuperlattice/blob/main/INTEGRATIONS.md) covers the conventions, each tool's workflow, and which adapter steps still need validation against an installed copy of that software.

## Verify the installation

Run the numerical regression tests from the repository root.

```sh
python -m unittest discover -v tests
```

The three scattering tests check the sphere form factor and FCC centering, diamond-silicon reflection conditions and powder invariance, and hexagonal ZnO spacings. The integration tests check exchange round trips, reflection comparisons, importer units, BornAgain geometry, atomistic assemblies, the Debye equation and symmetry classification. Tests that need Dans_Diffraction or BornAgain are skipped when those packages are absent.

## Scientific references

The silicon example uses the orientation relationship and FCC period of Yu et al., *Nano Letters* 16, 7814–7821 (2016). The ZnO atomic coordinates use the 295 K structural data of Kisi and Elcombe. The ZnO superlattice is a constructed example. [Silicon study](https://doi.org/10.1021/acs.nanolett.6b04006) · [ZnO structure](https://doi.org/10.1107/S0108270189004269)
