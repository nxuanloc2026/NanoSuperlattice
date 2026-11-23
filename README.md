# Nanocrystal superlattice visualizer

Generate an offline, interactive 3D view of a nanocrystal assembly and the atomic unit cell of a selected particle. The viewer can also simulate X-ray small-angle scattering (SAXS) and wide-angle scattering (WAXS) from the same material and orientations. Python and NumPy build the HTML file. A modern browser displays it without a server or internet connection.

## Run an example

With Python 3.9 or later, run these commands from the directory containing this file.

```sh
python -m pip install numpy
python visualize_superlattice.py examples/silicon_fcc.json -o silicon.html
python visualize_superlattice.py examples/hexagonal_zno.json -o zno.html
```

Open either HTML file in your browser. Click a particle or choose a variant in the atomic panel. Toggle **Particle axes** to draw that particle's direct-cell axes in the assembly. A magenta arrow in each panel shows the selected direction or plane normal. **Look along alignment** looks down that arrow. Drag either panel to rotate both, or use the arrow keys with a panel focused. Expand **NC → SL rotation matrix** to inspect the numeric transform.

In the silicon example, choose the first alignment and look along it. Then select the second alignment **without moving the camera**. Cycle through O1, O2, and O3. The [110] silicon direction stays aligned with [111] of the superlattice while the in-plane arrow turns through three azimuths. In the ZnO example, switch between the basal and inclined variants to see which atomic planes lie parallel to the assembly layers. Example site assignments are illustrative.

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
python visualize_superlattice.py material.json --scattering -o scattering.html
python visualize_superlattice.py material.json --no-scattering -o geometry.html
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

The bundled `f0_WaasKirf.dat` provides q-dependent X-ray scattering factors for tabulated neutral atoms and ions. For other species labels, configure `waxs.form_factors` as a mapping from label to a constant electron amplitude or to `{"a":[...],"b":[...],"c":...}` for f(q) = Σ aᵢ exp(−bᵢs²) + c, s = q/(40π) in Å⁻¹ when q is in nm⁻¹. [IUCr structure factor](https://dictionary.iucr.org/Structure_factor) · [Waasmaier & Kirfel](https://doi.org/10.1107/S0108767394013292) · [DABAX data](https://github.com/oasys-kit/DabaxFiles/blob/main/f0_WaasKirf.dat)

## Use from Python

```python
import json
from pathlib import Path
from visualize_superlattice import build_scene, write_viewer
from scattering import simulate_scattering

config = json.loads(Path("examples/hexagonal_zno.json").read_text(encoding="utf-8"))
scene = build_scene(config)
print(scene["variants"][1]["rotation"])
result = simulate_scattering(config, scene)
print(result["waxs"]["peaks"][0])  # q in nm⁻¹, d in nm
print(result["saxs"]["maxima"])    # local maxima of the slice-averaged curve
write_viewer(config, "zno.html", scattering=True)
```

The silicon example uses the orientation relationship and FCC period of Yu et al., *Nano Letters* 16, 7814–7821 (2016). The ZnO atomic coordinates use the 295 K structural data of Kisi and Elcombe. The ZnO superlattice is a constructed example. [Silicon study](https://doi.org/10.1021/acs.nanolett.6b04006) · [ZnO structure](https://doi.org/10.1107/S0108270189004269)
