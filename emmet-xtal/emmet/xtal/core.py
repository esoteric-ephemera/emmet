from __future__ import annotations

from collections import defaultdict
from functools import cached_property
from math import gcd
import re
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field, model_serializer, model_validator
from spglib import spglib

from emmet.xtal.typing import vector_3_t, matrix_3x3_t, list_vector_3_t, bool_3_t
from emmet.xtal.base import ShapeShifter, SETTINGS
from emmet.xtal.atom import Atom

try:
    from pymatgen.core import (
        Structure as PmgStructure,
        Composition as PmgComposition,
        Molecule as PmgMolecule,
    )
except ImportError:
    PmgStructure = None
    PmgComposition = None

try:
    from ase.atoms import Atoms as AseAtoms
except ImportError:
    AseAtoms = None

if TYPE_CHECKING:
    from collections.abc import KeysView, ValuesView, ItemsView, Sequence
    from typing import Any
    from typing_extensions import Self


class AtomProperties(BaseModel):
    """Define known properties of an Atom."""

    magmom: float | None = None
    charge: float | None = None
    selective_dynamics: bool_3_t | None = None
    velocity: vector_3_t | None = None

    def __hash__(self) -> int:
        return hash(tuple(getattr(self, k, None) for k in AtomProperties.model_fields))

    @model_serializer
    def unset_none(self):
        model_deser = {}
        for k in ("magmom", "selective_dynamics", "velocity"):
            if (v := getattr(self, k)) is not None:
                model_deser[k] = v
        return model_deser

    def items(self):
        return self.model_dump().items()

    def __bool__(self) -> bool:
        return any(
            getattr(self, k, None) is not None for k in AtomProperties.model_fields
        )


class CellVector3D(BaseModel):
    """Represent the cell vectors a, b, c."""

    matrix: matrix_3x3_t

    def __hash__(self) -> int:
        return hash(self.matrix)

    @property
    def volume(self) -> float:
        return abs(np.linalg.det(self.matrix))

    @cached_property
    def _reciprocal(self) -> np.ndarray:
        matrix = np.array(self.matrix)
        vol_fac = 1.0 / abs(np.dot(matrix[0], np.cross(matrix[1], matrix[2])))
        return vol_fac * np.array(
            [np.cross(matrix[(i + 1) % 3], matrix[(i + 2) % 3]) for i in range(3)]
        )

    @property
    def reciprocal(self) -> matrix_3x3_t:
        return tuple([tuple(2 * np.pi * v) for v in self._reciprocal])

    @cached_property
    def _vector_norms(self) -> vector_3_t:
        return tuple([np.linalg.norm(v) for v in self.matrix])

    @cached_property
    def _angles(self) -> vector_3_t:
        return [
            180
            / np.pi
            * np.arccos(
                np.dot(self.matrix[i], self.matrix[(i + 1) % 3])
                / (self._vector_norms[i] * self._vector_norms[(i + 1) % 3])
            )
            for i in range(3)
        ]

    @property
    def a(self) -> float:
        return self._vector_norms[0]

    @property
    def b(self) -> float:
        return self._vector_norms[1]

    @property
    def c(self) -> float:
        return self._vector_norms[2]

    @property
    def alpha(self) -> float:
        return self._angles[1]

    @property
    def beta(self) -> float:
        return self._angles[2]

    @property
    def gamma(self) -> float:
        return self._angles[0]

    def _qr_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        q, r = np.linalg.qr(self.matrix)
        if np.prod(np.diagonal(r)) < 0:
            return -q, -r
        return q, r

    @property
    def upper_triangular(self) -> CellVector3D:
        return CellVector3D(matrix=self._qr_decomposition()[1])


class Composition(ShapeShifter):
    """Basic interface for composition tools."""

    atoms: dict[Atom, float]

    def values(self) -> ValuesView:
        return self.atoms.values()

    def keys(self) -> KeysView:
        return self.atoms.keys()

    def items(self) -> ItemsView:
        return self.atoms.items()

    @model_validator(mode="before")
    @classmethod
    def serialize_from_dict(cls, config: Any) -> Any:
        if not config.get("atoms"):
            config["atoms"] = {k: v for k, v in config.items()}
        og_atoms = list(config["atoms"])
        for k in og_atoms:
            if isinstance(k, str):
                config["atoms"][Atom.from_str(k)] = config["atoms"].pop(k)
        return config

    @model_serializer()
    def to_dict(self) -> dict[str, float]:
        return {str(atom): occu for atom, occu in self.atoms.items()}

    @classmethod
    def _from_pymatgen(cls, composition: PmgComposition) -> Self:
        return cls(
            atoms={
                Atom._from_pymatgen(species): occu
                for species, occu in composition.items()
            }
        )

    def _to_pymatgen(
        self,
    ) -> PmgComposition:
        return PmgComposition(
            {atom._to_pymatgen(): occu for atom, occu in self.atoms.items()}
        )

    @classmethod
    def from_list(
        cls, atoms: list[Atom | dict[Atom, float] | Composition]
    ) -> Composition:
        comp = defaultdict(float)
        for atom in atoms:
            if isinstance(atom, Atom):
                comp[str(atom)] += 1
            elif isinstance(atom, dict | Composition):
                for sub_atom, occu in atom.items():
                    comp[str(sub_atom)] += occu
            else:
                raise ValueError("Cannot parse provided list of atoms.")
        return cls(atoms=comp)

    @cached_property
    def reduced_composition(self) -> Composition:
        rounded_comp = {k: round(v, None) for k, v in self.atoms.items()}
        if any(abs(v - self.atoms[k]) > 1.0e-6 for k, v in rounded_comp.items()):
            return self.atoms

        scl_fac = gcd(*rounded_comp.values())
        return Composition(atoms={k: v // scl_fac for k, v in rounded_comp.items()})

    @property
    def chemical_system(self) -> str:
        return "-".join(sorted(atom.name for atom in self.atoms))

    @staticmethod
    def _get_formula(comp: dict[str, float | int]) -> str:
        return " ".join(
            [f"{atom}{comp[atom]}" for atom in sorted(comp, key=lambda k: k.Z)]
        )

    @property
    def formula(self) -> str:
        return self._get_formula(self.atoms)

    @property
    def formula_reduced(self) -> str:
        return self._get_formula(self.reduced_composition.atoms)

    def __hash__(self) -> int:
        return hash(self.formula)


def set_coords(
    cell: CellVector3D, coords: list_vector_3_t, to: Literal["cartesian", "direct"]
) -> list_vector_3_t:
    if to == "direct":
        # return [
        #     tuple(v) for v in np.linalg.solve(cell.T, np.array(coords).T).T
        # ]
        return [
            tuple(v)
            for v in np.einsum("ij,ki->kj", cell._reciprocal.T, np.array(coords))
        ]
    elif to == "cartesian":
        return [
            tuple(v)
            for v in np.einsum("ij,ki->kj", np.array(cell.matrix), np.array(coords))
        ]
    raise ValueError(
        f'Unknown transformation {to}. Please select "cartesian" or "direct".'
    )


class NonPeriodicConfig(ShapeShifter):
    """Represent a set of atoms and their coordinates in space."""

    atoms: list[Atom]
    coords: list_vector_3_t
    atom_properties: list[AtomProperties] | None = None

    @property
    def num_sites(self) -> int:
        return len(self.atoms)

    @cached_property
    def composition(self) -> Composition:
        return Composition.from_list(self.atoms)

    @property
    def formula(self) -> str:
        return self.composition.formula

    @property
    def elements(self) -> list[str]:
        return sorted(self.composition)

    def center_of_mass(self) -> vector_3_t:
        masses = np.array(
            [
                (
                    sum(occu * atom.mass for atom, occu in site.items())
                    if isinstance(site, dict | Composition)
                    else site.mass
                )
                for site in self.atoms
            ]
        )
        return np.einsum("i,ij->j", masses, np.array(self.coords)) / masses.sum()

    @classmethod
    def _from_pymatgen(cls, atoms: PmgStructure | PmgMolecule, **kwargs) -> Self:

        config = {
            "atoms": [],
            "coords": [],
            "atom_properties": [],
        }
        for site in atoms:
            config["atoms"].append(Atom._from_pymatgen(site.species_string))
            config["coords"].append(site.coords)
            config["atom_properties"].append(AtomProperties(**site.properties))

        if not any(config["atom_properties"]):
            config["atom_properties"] = None

        return cls(**config, **kwargs)

    def _species(self) -> list[dict[str, float]]:
        """Aggregate atoms on each site into a list of dicts."""
        return [
            (
                {str(k): v for k, v in atom.items()}
                if isinstance(atom, dict | Composition)
                else {str(atom): 1.0}
            )
            for atom in self.atoms
        ]

    def _aggregate_site_properties(self) -> dict[str, list[Any]]:
        """Aggregate site properties into a pymatgen-like dict."""

        if self.atom_properties is None or not any(self.atom_properties):
            return {}

        # Important that the check here is only for non-null properties
        # as zero-valued properties are relevant
        non_null_props = [
            k
            for k in AtomProperties.model_fields
            if any(
                getattr(props, k, None) is not None for props in self.atom_properties
            )
        ]
        return {
            k: [getattr(props, k, None) for props in self.atom_properties]
            for k in non_null_props
        }

    def _to_pymatgen(
        self,
    ) -> PmgMolecule:
        return PmgMolecule(
            species=self._species(),
            coords=self.coords,
            site_properties=self._aggregate_site_properties(),
        )

    def _group_coords(
        self, ignore: Sequence[str] | None = None
    ) -> dict[Atom | Composition, np.ndarray]:

        ignore = set(ignore or [])
        allowed_props = set(Atom.model_fields).difference(ignore)
        groups = defaultdict(list)
        for idx, c in enumerate(self.atoms):
            if ignore:
                if isinstance(c, dict | Composition):
                    new_c = Composition(
                        atoms={
                            Atom(
                                **{k: getattr(atom, k, None) for k in allowed_props}
                            ): occu
                            for atom, occu in c.items()
                        }
                    )
                else:
                    new_c = Atom(**{k: getattr(c, k, None) for k in allowed_props})
            else:
                new_c = c.copy()
            groups[new_c].append(self.coords[idx])
        return {k: np.array(v) for k, v in groups.items()}


class PeriodicConfig(NonPeriodicConfig):
    """Represent a set of atoms with periodicity."""

    cell: CellVector3D
    pbc: bool_3_t = Field()

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.atoms),
                tuple(self.coords),
                self.cell,
                tuple(self.atom_properties),
            )
        )

    @cached_property
    def frac_coords(self) -> list_vector_3_t | None:
        if self.cell:
            return set_coords(self.cell, self.coords, "direct")
        return

    @property
    def volume(self) -> float:
        return self.cell.volume

    @property
    def density(self) -> float:
        """Structure density in g/cm^3."""
        return (
            sum(stoich * atom.mass for atom, stoich in self.composition.items())
            * 1e24
            / self.cell.volume
        )

    @classmethod
    def _from_pymatgen(cls, atoms: PmgStructure) -> Self:

        if not atoms.is_ordered:
            raise ValueError(
                "Please use `DisorderedConfig` to represent a disordered structure."
            )

        aux_config = {
            "cell": CellVector3D(matrix=atoms.lattice.matrix),
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
        """Create an spglib-compatible representation of the atoms."""
        return (
            self.cell.matrix,
            self.frac_coords,
            [atom.Z for atom in self.atoms],
        )

    @classmethod
    def _from_spglib(
        cls, spglib_rep: tuple[matrix_3x3_t, list_vector_3_t, list[int]]
    ) -> Self:
        cell, frac_coords, atomic_numbers = spglib_rep
        cell = CellVector3D(matrix=cell)
        return cls(
            atoms=[Atom.from_atomic_number(z) for z in atomic_numbers],
            cell=cell,
            coords=set_coords(cell, frac_coords, to="cartesian"),
            pbc=(True, True, True),
        )

    def primitive(
        self, symprec: float = SETTINGS.SYMPREC, angprec: float = SETTINGS.ANGPREC
    ) -> PeriodicConfig:
        return self._from_spglib(
            spglib.find_primitive(
                self._to_spglib, symprec=symprec, angle_tolerance=angprec
            )
        )

    def conventional(
        self, symprec: float = SETTINGS.SYMPREC, angprec: float = SETTINGS.ANGPREC
    ) -> PeriodicConfig:
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
        return tuple(re.match("(.*) \((.*)\)", sg_info).groups())

    def get_space_group_symbol(self):
        return self.get_space_group_info()[0]

    def get_space_group_number(self):
        return self.get_space_group_info()[1]

    def scale_volume(self, scale_factor: float) -> PeriodicConfig:

        npbc = len([v for v in self.pbc if v])
        per_cell_vector = scale_factor ** (1 / npbc)

        new_cell = CellVector3D(matrix=per_cell_vector * np.array(self.cell.matrix))

        new_cart_coords = set_coords(new_cell, self.frac_coords, to="cartesian")

        return type(self)(
            atoms=self.atoms,
            coords=new_cart_coords,
            cell=new_cell,
            pbc=self.pbc,
            atom_properties=self.atom_properties,
        )

    def standardized(self) -> PeriodicConfig:

        site_order = np.argsort([atom.Z for atom in self.atoms])

        new_cell = self.cell.upper_triangular
        new_direct_coords = np.array(
            set_coords(new_cell, [self.coords[idx] for idx in site_order], to="direct")
        )
        new_direct_coords = new_direct_coords - new_direct_coords[0]
        new_direct_coords = [[x % 1 for x in v] for v in new_direct_coords]

        new_cart_coords = set_coords(new_cell, new_direct_coords, to="cartesian")

        return PeriodicConfig(
            atoms=[self.atoms[idx] for idx in site_order],
            coords=new_cart_coords,
            cell=new_cell,
            pbc=self.pbc,
            atom_properties=(
                [self.atom_properties[idx] for idx in site_order]
                if self.atom_properties
                else None
            ),
        )


class DisorderedConfig(PeriodicConfig):
    """Represent a configurationally-disordered set of atoms."""

    atoms: list[Composition]

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
            "atoms": [],
            "coords": [],
            "cell": CellVector3D(matrix=atoms.lattice.matrix),
            "pbc": (True,) * 3,
            "frac_coords": [],
            "atom_properties": [],
        }
        for site in atoms:
            site_comp = Composition._from_pymatgen(site.species)
            config["atoms"].append(site_comp)
            if site_tol and abs(sum(site_comp.values()) - 1.0) > site_tol:
                raise ValueError(
                    f"Fractional site occupancy {sum(site_comp.values())} "
                    f"exceeds {site_tol} tolerance."
                )
            config["coords"].append(site.coords)
            config["frac_coords"].append(site.frac_coords)
            config["atom_properties"].append(AtomProperties(**site.properties))
        if not any(config["atom_properties"]):
            config["atom_properties"] = None

        return cls(**config)
