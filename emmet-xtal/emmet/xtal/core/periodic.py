
from __future__ import annotations

from functools import cached_property
import json
import numpy as np
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler
import re
import spglib
from typing import TYPE_CHECKING

from emmet.xtal.base import SETTINGS
from emmet.xtal.core.atom import AtomsMixIn, AtomSymbol
from emmet.xtal.core.cell import Cell
from emmet.xtal.core.base import AtomProperties, Composition

if TYPE_CHECKING:
    from typing_extensions import Self

def set_coords(
    cell: Cell, coords: np.ndarray, to: Literal["cartesian", "direct"]
) -> list_vector_3_t:
    if to == "direct":
        return np.einsum("ij,ki->kj", cell._reciprocal.T, coords)
    elif to == "cartesian":
        return np.einsum("ij,ki->kj", cell, coords)
    raise ValueError(
        f'Unknown transformation {to}. Please select "cartesian" or "direct".'
    )

class Material(AtomsMixIn):

    __slots__ = ["z", "cart_coords", "cell", "_frac_coords", "charges", "spins", "velocities", "degrees_of_freedom"]

    def __init__(
        self, 
        cell : list[list[float]],
        elements : list[int | str],
        coords : list[list[float]],
        coords_are_cartesian: bool = False,
        charges : list[float] | None = None,
        spins : list[float] | None = None,
        velocities : list[list[float]] | None = None,
        degrees_of_freedom : list[list[bool]] | None = None,
    ) -> None:

        self.cell = Cell(cell, dtype=float)

        num_sites = len(elements)
        
        self.z = np.array([Atom(ele).Z if isinstance(ele,str) else ele for ele in elements], dtype=int)

        arr_coords = np.array(coords, dtype=float)
        if not arr_coords.shape == (num_sites,3):
            raise ValueError(
                f"Misaligned shapes of `elements` ({num_sites}) and cartesian coordinates ({arr_coords.shape})"
            )
        if coords_are_cartesian:
            self.cart_coords = arr_coords
            self._frac_coords = None
        else:
            self._frac_coords = arr_coords
            self.cart_coords = set_coords(self.cell, arr_coords, to = "cartesian")

        self.charges, self.spins, self.velocities, self.degrees_of_freedom = self.parse_atom_properties(
            num_sites, charges, spins, velocities, degrees_of_freedom
        )

    def __repr__(self) -> str:
        return (
            f"{self.cell.__repr__()}\n"
            + "\n".join(
                f"{AtomSymbol.from_atomic_number(z).name}: {self.cart_coords[i]} (xyz) / {self.frac_coords[i]} (abc)"
                for i, z in enumerate(self.z)
            )
        )

    @classmethod
    def from_jsonable(cls, dct: dict ):
        return cls(
            *(
                dct.get(k) for k in ("cell", "elements","cart_coords",)
            ),
            coords_are_cartesian=True,
            **{
                k : dct.get(k)
                for k in ("charges","spins","velocities","degrees_of_freedom")
            }
        )

    @property
    def frac_coords(self) -> np.ndarray:
        if self._frac_coords is None:
            self._frac_coords = set_coords(self.cell, self.cart_coords, to = "direct")
        return self._frac_coords

    @property
    def volume(self) -> float:
        return self.cell.volume

    @property
    def density(self) -> float:
        """Structure density in g/cm^3."""
        return self._masses.sum() * 1e24 / self.cell.volume

    @cached_property
    def _to_spglib(self) -> tuple[Cell, np.ndarray, np.ndarray]:
        """Create an spglib-compatible representation of the atoms."""
        return (
            self.cell,
            self.frac_coords,
            self.z,
        )

    @classmethod
    def _from_spglib(
        cls, spglib_rep: tuple[Cell, np.ndarray, np.ndarray]
    ) -> Self:
        cell, frac_coords, atomic_numbers = spglib_rep
        return cls(
            cell,
            atomic_numbers,
            frac_coords,
            coords_are_cartesian=False,
        )

    def primitive(
        self, symprec: float = SETTINGS.SYMPREC, angprec: float = SETTINGS.ANGPREC
    ) -> Material:
        return self._from_spglib(
            spglib.find_primitive(
                self._to_spglib, symprec=symprec, angle_tolerance=angprec
            )
        )

    def conventional(
        self, symprec: float = SETTINGS.SYMPREC, angprec: float = SETTINGS.ANGPREC
    ) -> Material:
        return self._from_spglib(
            spglib.standardize_cell(
                self._to_spglib, symprec=symprec, angle_tolerance=angprec
            )
        )

    def get_space_group_info(
        self, symprec: float = SETTINGS.SYMPREC, angprec: float = SETTINGS.ANGPREC
    ) -> tuple[str, int]:
        sg_info = spglib.get_spacegroup(
            self._to_spglib, symprec=symprec, angle_tolerance=angprec
        )
        return tuple(re.match(r"(.*) \((.*)\)", sg_info).groups())

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda x : cls.from_jsonable(json.loads(x)),
            core_schema.json_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x : json.dumps(cls._to_jsonable(x)),
                info_arg=False,
                return_schema=core_schema.json_schema(),
            ),
        )

"""
class PydanticMaterial(PydanticMolecule):
    # Represent a set of atoms with periodicity.

    cell: Cell
    pbc: bool_3_t = Field()

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.atomic_numbers),
                tuple(self.coords),
                self.cell,
                tuple(self.atom_properties),
            )
        )

    @cached_property
    def frac_coords(self) -> list_vector_3_t:
        return set_coords(self.cell, self.coords, "direct")


    @classmethod
    def _from_pymatgen(cls, atoms: PmgStructure) -> Self:

        if not atoms.is_ordered:
            raise ValueError(
                "Please use `DisorderedMaterial` to represent a disordered structure."
            )

        aux_config = {
            "cell": Cell(atoms.lattice.matrix),
            "pbc": (True,) * 3,
        }
        return super()._from_pymatgen(atoms, **aux_config)

    def _to_pymatgen(self) -> PmgStructure:
        return PmgStructure(
            species=self._species(),
            lattice=self.cell.matrix,
            coords=self.coords,
            site_properties=self._aggregate_site_properties(),
            coords_are_cartesian=True,
        )

    @cached_property
    def _to_spglib(self) -> tuple[matrix_3x3_t, list_vector_3_t, list[int]]:
        # Create an spglib-compatible representation of the atoms.
        return (
            self.cell.matrix,
            self.frac_coords,
            self.atomic_numbers,
        )

    @classmethod
    def _from_spglib(
        cls, spglib_rep: tuple[matrix_3x3_t, list_vector_3_t, list[int]]
    ) -> Self:
        cell, frac_coords, atomic_numbers = spglib_rep
        cell = Cell(cell)
        return cls(
            atoms=atomic_numbers,
            cell=cell,
            coords=set_coords(cell, frac_coords, to="cartesian"),
            pbc=(True, True, True),
        )

    def _to_ase(self, **kwargs):
        return super()._to_ase(cell=self.cell.matrix, pbc=self.pbc, **kwargs)

    def primitive(
        self, symprec: float = SETTINGS.SYMPREC, angprec: float = SETTINGS.ANGPREC
    ) -> Material:
        return self._from_spglib(
            spglib.find_primitive(
                self._to_spglib, symprec=symprec, angle_tolerance=angprec
            )
        )

    def conventional(
        self, symprec: float = SETTINGS.SYMPREC, angprec: float = SETTINGS.ANGPREC
    ) -> Material:
        return self._from_spglib(
            spglib.standardize_cell(
                self._to_spglib, symprec=symprec, angle_tolerance=angprec
            )
        )

    def get_space_group_info(
        self, symprec: float = SETTINGS.SYMPREC, angprec: float = SETTINGS.ANGPREC
    ) -> tuple[str, int]:
        sg_info = spglib.get_spacegroup(
            self._to_spglib, symprec=symprec, angle_tolerance=angprec
        )
        return tuple(re.match(r"(.*) \((.*)\)", sg_info).groups())

    def get_space_group_symbol(self):
        return self.get_space_group_info()[0]

    def get_space_group_number(self):
        return self.get_space_group_info()[1]

    def scale_volume(self, scale_factor: float) -> Material:

        npbc = len([v for v in self.pbc if v])
        per_cell_vector = scale_factor ** (1 / npbc)

        new_cell = Cell(per_cell_vector * np.array(self.cell.matrix))

        new_cart_coords = set_coords(new_cell, self.frac_coords, to="cartesian")

        return type(self)(
            atomic_numbers=self.atomic_numbers,
            coords=new_cart_coords,
            cell=new_cell,
            pbc=self.pbc,
            atom_properties=self.atom_properties,
        )

    def standardized(self) -> Material:

        site_order = np.argsort(self.atomic_numbers)

        new_cell = self.cell.upper_triangular
        new_direct_coords = np.array(
            set_coords(new_cell, [self.coords[idx] for idx in site_order], to="direct")
        )
        new_direct_coords = new_direct_coords - new_direct_coords[0]
        new_direct_coords = [[x % 1 for x in v] for v in new_direct_coords]

        new_cart_coords = set_coords(new_cell, new_direct_coords, to="cartesian")

        return Material(
            atomic_numbers=self.atomic_numbers,
            coords=new_cart_coords,
            cell=new_cell,
            pbc=self.pbc,
            atom_properties=(
                [self.atom_properties[idx] for idx in site_order]
                if self.atom_properties
                else None
            ),
        )


class DisorderedMaterial(Material):
    # Represent a configurationally-disordered set of atoms.

    atomic_numbers: tuple[dict[int, float], ...]

    @model_validator(mode="before")
    @classmethod
    def serialize_composition(cls, config: Any) -> Any:
        for isite, site in enumerate(config["atoms"]):
            if isinstance(site, dict):
                config["atoms"][isite] = Composition(atoms=site)
        return config

    @classmethod
    def _from_pymatgen(cls, atoms: PmgStructure, site_tol: float | None = 1.0e-2):

        config = {
            "atomic_numbers": [],
            "coords": [],
            "cell": Cell(atoms.lattice.matrix),
            "pbc": (True,) * 3,
            "atom_properties": [],
        }
        for site in atoms:
            site_comp = Composition._from_pymatgen(site.species)
            config["atomic_numbers"].append({k.Z: v for k, v in site_comp.items()})
            if site_tol and abs(sum(site_comp.values()) - 1.0) > site_tol:
                raise ValueError(
                    f"Fractional site occupancy {sum(site_comp.values())} "
                    f"exceeds {site_tol} tolerance."
                )
            config["coords"].append(site.coords)
            config["atom_properties"].append(AtomProperties(**site.properties))
        if not any(config["atom_properties"]):
            config["atom_properties"] = None

        return cls(**config)
"""