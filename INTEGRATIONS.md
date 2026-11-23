# Integrating external software

The single module `integrations.py` connects NanoSuperlattice to external diffraction and scattering codes. One JSON configuration describes the material, particle geometry, site positions and orientations. Every adapter reads that same configuration. The core viewer does not import any of it, and NumPy is still the only required dependency.

Run commands from the repository root. `python integrations.py --help` lists every subcommand; `python integrations.py probe` prints the installed BornAgain and Dans_Diffraction APIs. The file is divided into sections (conventions, exchange, symmetry, results, dans, bornagain, atomistic, debye, scattersim, formats, probe, cli); the text below names those sections where it used to name separate files.

## Status

| Tool | What the adapter provides | Verified here | Still to validate on your installation |
|---|---|---|---|
| Exchange files | `structure.cif`, `particles.csv`, `orientations.json`, `run.json`, profile CSVs | Round trip on a triclinic cell with a nontrivial rotation and a laboratory frame | — |
| Dans_Diffraction | Reflection list compared by hkl, q, absences and \|F\|² ratios | Comparison logic, against a stand-in module | The three Dans calls, its scattering-factor and B conventions |
| CrystalMaker / SingleCrystal | CIF and orientation matrices with an explicit convention | CIF reader and writer | Manual import; commercial licensing |
| BornAgain | Generated script and parameters, a detector q map, and a Born reference on the same pixels | Euler angles, detector geometry, sphere reference, shape volumes, script syntax | Class names, form-factor origins, local axes and result-array orientation, all by the sphere baseline |
| AES-Debye | XYZ coordinates, a job description and a command-template runner | Runner loop with a stand-in executable | The tool's own command-line options |
| DebUsSy | CIF, particle model, measured q/I/σ, and fitted values mapped back to a configuration | Bundle writer and `apply-fit` | Building the cluster database in CLAUDE, following the DebUsSy manual |
| ScatterSim | Parameter file, plus matching NanoSuperlattice powder and in-slice SAXS curves | Powder average against the analytic sphere | Translation into the ScatterSim API |
| DISCUS | Periodic-cell and finite-cluster structure files | File layout | Keyword set for your DISCUS version |
| Debye equation (built in) | Exact and histogram NumPy reference | Two-atom analytic case; Si 111, 220 and 311 positions; histogram accuracy | — |

`python -m unittest -v test_integrations` runs the 27 integration checks. The live Dans_Diffraction and BornAgain tests are skipped until those packages are installed, and then run automatically.

## Shared conventions

- Lengths are in nm, angles in degrees, and q in rad/nm, with q = 4π sin θ / λ = 2π/d. CIF files use Å; the exporters convert by a factor of 10.
- A holds the direct lattice vectors as rows. A fractional row f maps to x = f A. A direction [uvw] maps to Aᵀ[u,v,w]ᵀ. A plane normal (hkl) maps to A⁻¹[h,k,l]ᵀ, and a reciprocal vector is G = 2π A⁻¹ h.
- Particle rotations are active: x_SL = R x_NC and G_SL = R G_NC. Atom x of particle j sits at t_j + R_j x; for row arrays, use `x @ R.T + t`.
- The optional `laboratory` block defines a separate proper rotation U, with x_lab = U x_SL. It is never folded into a particle rotation.

```json
"laboratory": {"z": {"plane": [0, 0, 1]}, "x": {"direction": [1, 0, 0]}}
```

Lab z follows the first SL axis exactly. Lab x is the part of the second axis perpendicular to z. You can also give `{"matrix": U}`. The default is U = I. For BornAgain GISAXS, set z along the substrate normal and x along the in-plane beam direction.

### Displacement parameters

`b_iso_nm2` is the Debye–Waller B in nm², applied as exp(−B s²) with s = sin θ/λ. In a CIF it is written as U_iso in Å², using U = 100 B / (8π²).

### Importing q in other units

The importer converts q from 1/Å (×10) and from cycles per length (×2π). It converts 2θ with q = 4π sin θ / λ, which needs `wavelength_nm` in run.json. Raw intensity is always kept; display normalization happens only in the viewer.

## Exchange files and round trip

```sh
python integrations.py export examples/silicon_fcc.json exchange/si
python integrations.py roundtrip examples/hexagonal_zno.json exchange/zno
```

| File | Contents |
|---|---|
| `structure.cif` | P1 cell in Å with unique sites, type symbol, occupancy and U_iso. The reader also expands `_symmetry_equiv_pos_as_xyz` operators, so database CIFs can be imported. |
| `particles.csv` | Site ID, cell indices, basis site, center x/y/z in nm, variant ID and variant name |
| `orientations.json` | Each 3×3 matrix with its convention, frame definitions, both cell matrices, U, site counts and population fractions |
| `scattering_*.csv` | q, intensity and uncertainty, each with a `.run.json` sidecar that records the regime and averaging |
| `waxs_reflections.csv` | hkl, q, d, 2θ, multiplicity and Σ\|F\|² |
| `run.json` | Backend, version (git describe), config SHA-256, wavelength, axes, units, normalization and settings |

`roundtrip` reloads geometry from the CIF, CSV and JSON files alone. It compares cells, atoms, centers, rotations, U, and lab-frame direct and reciprocal vectors for several test indices. It exits nonzero if any difference exceeds 1e-7.

## Dans_Diffraction

```sh
python -m pip install Dans_Diffraction
python integrations.py dans examples/silicon_fcc.json --qmax 45 --json dans_si.json
python integrations.py dans examples/hexagonal_zno.json --qmax 40
```

The adapter generates hkl itself using A, so Dans' Cartesian frame never enters the comparison. It calls only `Crystal(cif)`, `Scatter.setup_scatter(scattering_type="xray", energy_kev=…)`, `Scatter.intensity(hkl)` and `Cell.Qmag(hkl)`. The comparison runs in order: q per hkl, then absences, then \|F\|² normalized to the strongest reflection. Expected results:

- **Silicon:** 111, 220 and 311 at about 20.04, 32.72 and 38.37 nm⁻¹; 200 and 222 absent.
- **ZnO:** a q mismatch here usually means a plane normal was converted with Aᵀ instead of A⁻¹. The test suite checks that this error is caught.

If intensity ratios differ, check that both codes use the same f0 table, no anomalous terms, and the same B convention. Pass extra `setup_scatter` options through `run_comparison(..., setup_kwargs=…)`. `oriented_spots` rotates the reflections with each R and U, and weights them by site fractions as in waxs.py.

## CrystalMaker / SingleCrystal

1. Open `structure.cif`.
2. Orient the crystal with the two alignment pairs listed under each variant in `orientations.json`. Alternatively, enter the matrix there, noting that it is active and acts on column vectors.
3. Set the beam from U and the selected SL viewing axis, and match the wavelength.
4. Export one pattern per variant, keeping `population_fraction` as its weight.
5. To overlay a numerical reflection export, save it as CSV with a `run.json` that names its columns and units, then pass it to `--overlay`.

## BornAgain

```sh
python integrations.py bornagain examples/silicon_fcc.json ba/si --experiment examples/experiment_gisaxs.json --repeat 1 1 1
python ba/si/bornagain_run.py                     # in an environment with BornAgain
python integrations.py compare-detector examples/silicon_fcc.json ba/si/bornagain_result.csv --experiment examples/experiment_gisaxs.json --repeat 1 1 1
python visualize_superlattice.py examples/silicon_fcc.json --overlay ba/si/bornagain_result.csv
```

The `experiment` block sets geometry (`gisaxs` or `transmission`), wavelength, incidence angle, detector (pixels, pixel size, distance, beam center), media (δ, β) and `born_baseline`. With `born_baseline: true`, the substrate equals the ambient medium, so BornAgain's DWBA reduces to the Born approximation that `detector_reference` evaluates. If the particle δ is not given, it is estimated from the f0(0) electron density. β must be supplied, because no f″ table is bundled.

Every particle becomes one component of a single coherent compound, at U t_j with rotation U R_j S, where S is an optional `bornagain.shape_frame` correction. Size conversions from the center-to-vertex radius r:

| NanoSuperlattice | BornAgain | Arguments | Status |
|---|---|---|---|
| sphere | Sphere / FormFactorFullSphere | r | |
| cube | Box | edge 2r/√3 | |
| octahedron | PlatonicOctahedron | edge √2 r | verify origin and axes |
| cuboctahedron | CantellatedCube | L = √2 r, removed length L/2 | verify the removed-length definition. BornAgain's `Cuboctahedron` is a different solid (a truncated bipyramid). |
| hexagonal_prism | Prism6 | edge = height = r/√1.25 | verify vertex azimuth |

The generated script picks class names that exist in the installed version, such as `RefractiveMaterial` or `HomogeneousMaterial`, and `ScatteringSimulation` or `GISASSimulation`. It also checks that `RotationEuler` reproduces each matrix. Validate in this order:

1. One sphere in a homogeneous medium.
2. A small periodic array.
3. Refraction, substrate reflection and instrumental effects.

Use `compare-detector` at each step. It reports the log correlation, the median ratio, the q-grid mismatch and the brightest pixel in each image. A flipped or transposed image shows up as a mismatch in the brightest pixel.

## Finite atomistic assemblies, AES-Debye and DebUsSy

```sh
python integrations.py atomistic examples/silicon_fcc.json atoms/si --estimate-only
python integrations.py atomistic examples/silicon_fcc.json atoms/si --repeat 1 1 1 --radius 2
python integrations.py debye examples/silicon_fcc.json debye/si.csv --repeat 1 1 1 --radius 2 --bin 0.0005 --isolated
python integrations.py compare-profiles debye/si.csv debye/si_isolated.csv
```

**Termination.** The fractional point `origin_frac` of the atomic lattice (default [0,0,0]) sits at the particle center. An atom is kept if it lies within 1e-9 nm of the envelope or inside it. Spheres use the exact radius. Other built-in shapes use their face planes, and nonconvex custom meshes use a winding-number test.

**Checks.** Each build reports atom counts against the volume estimate, duplicate atoms, and the occupancy sum. It also reports the closest atoms between particles whose bounding spheres touch, flagging overlap below 90 % of the crystal's nearest-neighbour distance. `--estimate-only` prints the atom and pair counts before anything is built. The full silicon example has about 4×10⁵ atoms and 9×10¹⁰ Debye pairs, so start from `--repeat 1 1 1` and a small radius.

**Disorder.** `--position-sigma` (nm) and `--rotation-sigma` (degrees) add seeded Gaussian disorder.

**Debye reference.** The debye section of `integrations.py` evaluates I(q) = Σ w_a f_a² + 2 Σ_{a<b} w_a w_b f_a f_b sin(q r_ab)/(q r_ab), including pairs between particles. `--isolated` also writes the sum of isolated particles, which is the limit of incoherent addition at wide angles. The histogram mode (`--bin`) agrees with the exact sum to about 1e-5 of the peak intensity at a 0.1 pm bin.

**AES-Debye.**

```sh
python integrations.py aes-debye examples/silicon_fcc.json aes/si --repeat 1 1 1 --radius 2 \
    --command "aes-debye {xyz} --q {qgrid} -o {output}"
```

This writes `assembly.xyz` (extended XYZ in Å, with occupancy and particle columns), `job.json` (q grid in nm⁻¹ and Å⁻¹, form-factor source, counts) and the NumPy reference curve. The command template shown is only an example: substitute the documented options of your installed AES-Debye. The output is read as `q (1/A),intensity` CSV, or any header the importer recognizes, and compared with the reference automatically.

**DebUsSy.**

```sh
python integrations.py debussy examples/silicon_fcc.json debussy/si --measured measured.csv
python integrations.py apply-fit examples/silicon_fcc.json fit.json refined.json
python visualize_superlattice.py refined.json -o refined.html
```

The bundle holds the CIF, the particle model and `measured.dat` (q in Å⁻¹, I, σ). If no σ is supplied, it writes √I and labels it as a placeholder. Build the cluster database with CLAUDE as the DebUsSy manual describes. `fit.json` accepts `cell_nm` or `cell_scale`, `radius_nm`, `superlattice_scale`, `b_iso_nm2` and `occupancy` per species, plus `fit_info`, which is stored under `provenance`. Keep the residuals and the cluster-generation settings in `fit_info`.

## ScatterSim and DISCUS

```sh
python integrations.py scattersim examples/silicon_fcc.json scattersim/si --qmax 1.6
```

This writes the lattice, particle and orientation parameters, along with two NanoSuperlattice curves:

- the 3D orientational (powder) mean, which corresponds to a circularly averaged isotropic powder;
- the in-slice azimuthal mean that the viewer shows.

These differ for any anisotropic assembly, so compare ScatterSim with the powder curve. Start with a sphere lattice, then move to an anisotropic particle.

The `atomistic` command also writes `assembly.discus`, a finite cluster in a 1 Å orthonormal cell with Cartesian Å coordinates and B in Å². `discus_cell_text(config["atomic"])` writes the periodic cell instead.

## Symmetry operations versus physical domains

```sh
python integrations.py symmetry examples/silicon_fcc.json
```

For each pair of variants, Δ = R_iᵀ R_j is tested against three groups:

- the proper lattice rotations, which leave spot positions unchanged;
- the crystal rotations, which also map the basis onto itself (24 for diamond Si, 6 for wurtzite);
- the envelope rotations, which map the particle mesh onto itself.

The command also reports the disorientation angle after factoring out crystal symmetry. The three silicon variants come out as physically distinct, with a 60° disorientation. A variant multiplied by a cubic 4-fold is identical for a cuboctahedron. For a hexagonal prism, the envelope distinguishes it.

## Viewer overlays

```sh
python visualize_superlattice.py examples/silicon_fcc.json -o si.html \
    --overlay debye/si.csv --overlay ba/si/bornagain_result.csv
```

Profiles are drawn over the SAXS or WAXS model profile of the same regime. By default each is scaled to the model with one log-space factor, fitted where the model exceeds 0.1 % of its maximum; untick the box to scale each curve to its own maximum. Detector images appear in an **Imported results** panel with their own pixel axes, a q readout and a provenance table. They are never resampled onto the central slice.

Each integration is complete when it has a reproducible export, a documented run, a numerical import and an agreed reference comparison. Record any remaining difference in physics before using external results to infer experimental structure.
