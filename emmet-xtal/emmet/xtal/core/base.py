from __future__ import annotations

from collections import defaultdict
from functools import cached_property
from math import gcd
import re
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field, model_serializer, model_validator

from emmet.xtal.base import ShapeShifter, SETTINGS
from emmet.xtal.core.atom import Atom
from emmet.xtal.typing import vector_3_t, matrix_3x3_t, list_vector_3_t, bool_3_t

try:
    from pymatgen.core import (
        Structure as PmgStructure,
        Composition as PmgComposition,
        Lattice as PmgLattice,
        Molecule as PmgMolecule,
    )
except ImportError:
    PmgStructure = None
    PmgComposition = None
    PmgLattice = None

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
        for k in ("magmom", "charge", "selective_dynamics", "velocity"):
            if (v := getattr(self, k)) is not None:
                model_deser[k] = v
        return model_deser

    def items(self):
        return self.model_dump().items()

    def __bool__(self) -> bool:
        return any(
            getattr(self, k, None) is not None for k in AtomProperties.model_fields
        )
        
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

    @property
    def mass_amu(self) -> float:
        return sum(stoich * atom.mass for atom, stoich in self.atoms.items())

    @classmethod
    def from_list(
        cls, atoms: list[Atom | int | str | dict[Atom | int | str, float] | Composition]
    ) -> Composition:

        comp = defaultdict(float)

        def from_symb(
            atom: int | Atom,
            occu: float = 1.0,
        ) -> None:
            if isinstance(atom, int):
                symb = Atom.from_atomic_number(atom)
            elif isinstance(atom, str):
                symb = Atom(atom)
            else:
                symb = atom
            comp[symb] += occu

        for atom in atoms:
            if isinstance(atom, dict | Composition):
                for sub_atom, occu in atom.items():
                    from_symb(sub_atom, occu=occu)
            elif isinstance(atom, int | Atom):
                from_symb(atom)
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
            [
                f"{atom}{comp[atom]}"
                for atom in sorted(comp, key=lambda k: k.atomic_number)
            ]
        )

    @property
    def formula(self) -> str:
        return self._get_formula(self.atoms)

    @property
    def formula_reduced(self) -> str:
        return self._get_formula(self.reduced_composition.atoms)

    def __hash__(self) -> int:
        return hash(self.formula)

"""
class PydanticMolecule(ShapeShifter):
    #Represent a set of atoms and their coordinates in space.

    atomic_numbers: tuple[int, ...]
    coords: list_vector_3_t
    atom_properties: tuple[AtomProperties, ...] | None = None

    @property
    def num_sites(self) -> int:
        return len(self.atomic_numbers)

    @cached_property
    def composition(self) -> Composition:
        return Composition.from_list(self.atomic_numbers)

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
                    Composition(atoms=site).mass_amu
                    if isinstance(site, dict | Composition)
                    else Atom.from_atomic_number(site).mass_amu
                )
                for site in self.atomic_numbers
            ]
        )
        return np.einsum("i,ij->j", masses, np.array(self.coords)) / masses.sum()

    @classmethod
    def _from_pymatgen(cls, atoms: PmgStructure | PmgMolecule, **kwargs) -> Self:

        config = {
            "atomic_numbers": [],
            "coords": [],
            "atom_properties": [],
        }
        for site in atoms:
            atom = Atom._from_pymatgen(site.species_string)
            config["atomic_numbers"].append(atom.atomic_number)
            config["coords"].append(site.coords)
            config["atom_properties"].append(
                AtomProperties(**site.properties, charge=atom.charge)
            )

        if not any(config["atom_properties"]):
            config["atom_properties"] = None

        return cls(**config, **kwargs)

    def _species(self) -> list[dict[str, float]]:
        #Aggregate atoms on each site into a list of dicts.
        return [
            (
                {str(Atom.from_atomic_number(k)): v for k, v in atom.items()}
                if isinstance(atom, dict | Composition)
                else {str(Atom.from_atomic_number(atom)): 1.0}
            )
            for atom in self.atomic_numbers
        ]

    def _aggregate_site_properties(self) -> dict[str, list[Any]]:
        #Aggregate site properties into a pymatgen-like dict.

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
        for idx, c in enumerate(self.atomic_numbers):
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

    def _to_ase(self, **kwargs) -> AseAtoms:

        props = {
            k: [getattr(prop, k) for prop in self.atom_properties]
            for k in (
                "magmom",
                "charge",
                "velocity",
            )
        }
        for k, v in props.items():
            if not all(x is not None for x in v):
                props[k] = None

        remap = {
            "magmom": "magmoms",
            "charge": "charges",
            "velocity": "velocities",
        }

        config = {
            "positions": self.coords,
            "numbers": self.atomic_numbers,
            "masses": [
                Atom.from_atomic_number(num).mass_amu for num in self.atomic_numbers
            ],
            **{v: props[k] for k, v in remap.items()},
            **kwargs,
        }
        return AseAtoms(
            **{
                k: np.array(v) if isinstance(v, list | tuple) else v
                for k, v in config.items()
            }
        )
"""