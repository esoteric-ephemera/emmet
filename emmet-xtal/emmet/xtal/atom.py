from __future__ import annotations

from enum import Enum
from pathlib import Path
import re
import requests
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import PrivateAttr
from scipy.constants import physical_constants

from emmet.xtal.base import PackageInterface, INSTALLED_PACKAGES, ShapeShifter
from emmet.xtal.typing import null_t

if INSTALLED_PACKAGES[PackageInterface.PMG]:
    from pymatgen.core import (
        Element as PmgElement,
        Species as PmgSpecies,
    )
else:
    PmgElement = null_t
    PmgSpecies = null_t

if TYPE_CHECKING:
    from typing_extensions import Self

ATOM_DATA_FILE = Path(__file__).absolute().parent / "data" / "isotope_data.json.gz"

if ATOM_DATA_FILE.exists():
    ATOM_DATA = pd.read_json(ATOM_DATA_FILE)
else:
    ATOM_DATA = None


class AtomSymbol(Enum):
    """Chemical element / isotope to longer American English name."""

    H = "Hydrogen"
    D = "Deuterium"
    T = "Tritium"
    He = "Helium"
    Li = "Lithium"
    Be = "Beryllium"
    B = "Boron"
    C = "Carbon"
    N = "Nitrogen"
    O = "Oxygen"
    F = "Fluorine"
    Ne = "Neon"
    Na = "Sodium"
    Mg = "Magnesium"
    Al = "Aluminum"
    Si = "Silicon"
    P = "Phosphorus"
    S = "Sulfur"
    Cl = "Chlorine"
    Ar = "Argon"
    K = "Potassium"
    Ca = "Calcium"
    Sc = "Scandium"
    Ti = "Titanium"
    V = "Vanadium"
    Cr = "Chromium"
    Mn = "Manganese"
    Fe = "Iron"
    Co = "Cobalt"
    Ni = "Nickel"
    Cu = "Copper"
    Zn = "Zinc"
    Ga = "Gallium"
    Ge = "Germanium"
    As = "Arsenic"
    Se = "Selenium"
    Br = "Bromine"
    Kr = "Krypton"
    Rb = "Rubidium"
    Sr = "Strontium"
    Y = "Yttrium"
    Zr = "Zirconium"
    Nb = "Niobium"
    Mo = "Molybdenum"
    Tc = "Technetium"
    Ru = "Ruthenium"
    Rh = "Rhodium"
    Pd = "Palladium"
    Ag = "Silver"
    Cd = "Cadmium"
    In = "Indium"
    Sn = "Tin"
    Sb = "Antimony"
    Te = "Tellurium"
    I = "Iodine"
    Xe = "Xenon"
    Cs = "Cesium"
    Ba = "Barium"
    La = "Lanthanum"
    Ce = "Cerium"
    Pr = "Praseodymium"
    Nd = "Neodymium"
    Pm = "Promethium"
    Sm = "Samarium"
    Eu = "Europium"
    Gd = "Gadolinium"
    Tb = "Terbium"
    Dy = "Dysprosium"
    Ho = "Holmium"
    Er = "Erbium"
    Tm = "Thulium"
    Yb = "Ytterbium"
    Lu = "Lutetium"
    Hf = "Hafnium"
    Ta = "Tantalum"
    W = "Tungsten"
    Re = "Rhenium"
    Os = "Osmium"
    Ir = "Iridium"
    Pt = "Platinum"
    Au = "Gold"
    Hg = "Mercury"
    Tl = "Thallium"
    Pb = "Lead"
    Bi = "Bismuth"
    Po = "Polonium"
    At = "Astatine"
    Rn = "Radon"
    Fr = "Francium"
    Ra = "Radium"
    Ac = "Actinium"
    Th = "Thorium"
    Pa = "Protactinium"
    U = "Uranium"
    Np = "Neptunium"
    Pu = "Plutonium"
    Am = "Americium"
    Cm = "Curium"
    Bk = "Berkelium"
    Cf = "Californium"
    Es = "Einsteinium"
    Fm = "Fermium"
    Md = "Mendelevium"
    No = "Nobelium"
    Lr = "Lawrencium"
    Rf = "Rutherfordium"
    Db = "Dubnium"
    Sg = "Seaborgium"
    Bh = "Bohrium"
    Hs = "Hassium"
    Mt = "Meitnerium"
    Ds = "Darmstadtium"
    Rg = "Roentgenium"
    Cn = "Copernicium"
    Nh = "Nihonium"
    Fl = "Flerovium"
    Mc = "Moscovium"
    Lv = "Livermorium"
    Ts = "Tennessine"
    Og = "Oganesson"


NAMED_ISOTOPES = [
    "D",
    "T",
]


def get_value_and_uncertainty(val_str: str) -> tuple[float, float | None]:
    reg = re.match("([0-9]+)*(\.[0-9]+)?(\([0-9]+\))?", val_str)
    if not reg or not any(reg.groups()):
        raise ValueError(f"Malformed input string {val_str}.")
    characteristic, mantissa, uncertainty = reg.groups()
    characteristic = float(characteristic or 0)

    min_pow = len(mantissa) - 1 if mantissa else 0
    mantissa = float(mantissa or 0)

    if uncertainty:
        uncertainty = 10 ** (-min_pow) * float(
            uncertainty.replace("(", "").replace(")", "")
        )
    return characteristic + mantissa, uncertainty


def fetch_isotope_data(
    url: str = "https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii2&isotype=some",
    data_file_name: str | Path | None = ATOM_DATA_FILE,
) -> pd.DataFrame:
    """Parse neutral isotope data from NIST.

    Parameters:
    ------------
    url : str
        The NIST URL storing isotope data in ASCII format.
    data_file_name : str or Path or None
        If a str or Path, the name of the file to write the data to.
        If None, does not write a file.

    Returns:
    ------------
    pandas DataFrame containing isotope properties.
    """

    recognized_keys: dict[str, str] = {
        "Atomic Number": "Z",
        "Atomic Symbol": "name",
        "Mass Number": "A",
        "Relative Atomic Mass": "mass_amu",
        "Isotopic Composition": "relative_abundance",
        # "Standard Atomic Weight": "mass_amu"
    }

    resp = requests.get(url)
    if not resp.status_code == 200:
        raise ValueError(
            f"Could not connect to {url}, received status code {resp.status_code}."
        )

    iso_data = {}
    entry = {}
    atomic_number: int | None = None
    for line in resp.content.decode().splitlines():
        vs = [c.strip() for c in line.split("=")]
        if vs and (known_key := recognized_keys.get(vs[0])):

            if vs[0] == "Atomic Number":

                if atomic_number:
                    if atomic_number not in iso_data:
                        iso_data[atomic_number] = []
                    iso_data[atomic_number].append(entry)

                atomic_number = int(vs[1])
                entry = {}

            else:

                if known_key in ("Z", "A"):
                    entry[known_key] = int(vs[1])
                elif known_key == "name":
                    entry[known_key] = vs[1]
                    entry["long_name"] = AtomSymbol[vs[1]].value
                elif known_key == "mass_amu" and "[" in vs[1]:
                    vals = vs[1].replace("[", "").replace("]", "").split(",")
                    entry["mass_amu"] = sum(
                        get_value_and_uncertainty(v)[0] for v in vals
                    ) / len(vals)
                else:
                    try:
                        entry[known_key] = get_value_and_uncertainty(vs[1])[0]
                    except ValueError:
                        entry[known_key] = None if "&nbsp" in vs[1] else vs[1]

    columns = ["name", "long_name", "Z", "A", "mass_amu", "primary_isotope"]
    most_abundant_iso = []
    for z, entries in iso_data.items():
        by_name = {
            name: {"relative_abundance": 0.0}
            for name in set([entry["name"] for entry in entries])
        }
        for entry in entries:
            entry["relative_abundance"] = (
                entry["relative_abundance"]
                if isinstance(entry["relative_abundance"], float)
                else 1.0
            )
            if (
                entry["relative_abundance"]
                > by_name[entry["name"]]["relative_abundance"]
            ):
                by_name[entry["name"]].update(
                    **entry, Z=z, primary_isotope=(entry["name"] not in NAMED_ISOTOPES)
                )

        for entry in by_name.values():
            most_abundant_iso.append({k: entry[k] for k in columns})

    data = pd.DataFrame(
        most_abundant_iso,
        columns=columns,
    )
    if data_file_name:
        data.to_json(data_file_name)

    return data


class Atom(ShapeShifter):
    """Basic representation of an atom."""

    name: str
    charge: float | None = None
    spin: float | None = None
    _long_name: str | None = PrivateAttr(None)
    _mass_amu: float | None = PrivateAttr(None)

    def __hash__(self) -> int:
        """Hash based on properties."""
        return hash((self.Z, self.charge, self.spin, self.long_name, self.mass_amu))

    @classmethod
    def from_str(cls, rep: str, **kwargs) -> Self:
        """Parse an Atom from a string, including possible oxieation state."""
        _parsed = re.match("([A-Z][a-z]?)([0-9.0-9]+)?([+-])?", rep)
        if not _parsed:
            raise ValueError(f"Unknown element symbol {rep}")
        parsed = _parsed.groups()
        charge_str = parsed[1]

        charge_sign = "+"
        if parsed[2] is not None:
            charge_sign = parsed[2]
            if charge_str is None:
                charge_str = "1"

        config = {
            "name": parsed[0],
            "charge": kwargs.pop("charge", None),
            **kwargs,
        }
        if charge_str is not None:
            config["charge"] = float(charge_sign + charge_str)

        ref_atom_data = ATOM_DATA[ATOM_DATA.name == config["name"]]
        for k in [*cls.model_fields.keys()]:
            if k not in config and k in ref_atom_data:
                config[k] = ref_atom_data[k].squeeze()

        obj = cls(**config)

        for k in ("long_name", "mass_amu"):
            if v := kwargs.pop(k, None):
                setattr(obj, f"_{k}", v)

        return obj

    @classmethod
    def from_atomic_number(cls, z: int, **kwargs):
        """Create an Atom from the proton number."""
        mask = ATOM_DATA.Z == z
        for k, v in kwargs.items():
            if ATOM_DATA.get(k):
                mask = mask & ATOM_DATA[ATOM_DATA[k] == v]

        if len(kwargs) == 0 or len(ATOM_DATA[mask]) > 0:
            mask = mask & ATOM_DATA.primary_isotope

        return cls.from_str(
            ATOM_DATA[mask].name.squeeze(),
            **kwargs,
        )

    @classmethod
    def _from_pymatgen(cls, species: PmgElement | PmgSpecies):
        """Create a pymatgen Element or Species."""
        kwargs = {}
        if spin := getattr(species, "spin", None):
            kwargs["spin"] = spin
        return cls.from_str(str(species), **kwargs)

    def _to_pymatgen(
        self,
    ) -> PmgElement | PmgSpecies:
        """Create an Atom from a pymatgen Element or Species."""
        if self.charge or self.spin:
            return PmgSpecies(
                self.name,
                oxidation_state=self.charge,
                spin=self.spin,
            )
        return PmgElement(self.name)

    @property
    def Z(self) -> int:
        """Return the number of protons in the atom."""
        return ATOM_DATA[ATOM_DATA.name == self.name].Z.squeeze()

    @property
    def mass_amu(self) -> float:
        """Get the atom mass in atomic mass units (amu)."""
        if not self._mass_amu:
            self._mass_amu = ATOM_DATA[ATOM_DATA.name == self.name].mass_amu.squeeze()
        return self._mass_amu

    @property
    def long_name(self) -> str:
        """Return the full name of the atom, e.g., Hydrogen."""
        if not self._long_name:
            self._long_name = AtomSymbol[self.name]
        return self._long_name

    @property
    def mass(self) -> float:
        """Get the atom mass in gram."""
        return (
            self.mass_amu
            * physical_constants["atomic mass unit-kilogram relationship"][0]
            * 1e3
        )

    def __str__(self) -> str:
        """The element with its optional charge/oxidation state."""
        charge_sign = ""
        chg_str = ""
        if self.charge is not None:
            # specifically want to check if charge is set and zero here
            if self.charge > 0:
                charge_sign = "+"
            elif self.charge < 0:
                charge_sign = "-"
            chg_str = f"{abs(self.charge)}"
        return f"{self.name}{chg_str}{charge_sign}"
