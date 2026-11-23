"""NanoSuperlattice: oriented nanocrystal superlattices and their X-ray scattering.

Command-line tools (after ``pip install nanosuperlattice``):

    nanosuperlattice examples/silicon_fcc.json -o silicon.html
    nanosuperlattice-integrations --help

The same tools run without installing, from a checkout of the repository:

    python -m nanosuperlattice examples/silicon_fcc.json -o silicon.html
    python -m nanosuperlattice.integrations --help

NumPy is the only required dependency. BornAgain and Dans_Diffraction are
optional and are imported only by the integration commands that need them.
"""

__version__ = "0.1.0"

from .crystal_geometry import lattice_matrix
from .scattering import simulate_scattering
from .visualize_superlattice import build_scene, write_viewer

__all__ = ["__version__", "build_scene", "lattice_matrix", "simulate_scattering", "write_viewer"]
