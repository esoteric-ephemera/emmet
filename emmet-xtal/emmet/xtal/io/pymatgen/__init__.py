"""Define backwards-compatible imports and interfaces to pymatgen."""

from emmet.xtal.base import INSTALLED_PACKAGES, PackageInterface

if not INSTALLED_PACKAGES[PackageInterface.PMG]:
    raise ImportError("`pymatgen` must be installed to use `emmet.xtal.io.pymatgen`")