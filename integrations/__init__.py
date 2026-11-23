"""Optional bridges between NanoSuperlattice and external scattering software.

Nothing here is imported by the core viewer. External packages are imported
lazily inside the adapter that needs them, so the core keeps NumPy as its only
dependency. Run ``python -m integrations.cli --help`` from the repository root.

Shared conventions (see integrations/conventions.py):
  * lengths in nm, angles in degrees, q in rad/nm (q = 2*pi/d);
  * cell matrices A hold direct basis vectors in rows, x = f @ A;
  * particle rotations are active, x_SL = R @ x_NC;
  * a separate laboratory rotation maps the SL frame to the lab, x_lab = U @ x_SL.
"""

FORMAT_VERSION = "1.0"
