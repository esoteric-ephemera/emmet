"""Tools for parsing normal, extended, and trajectory xyz files."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from emmet.xtal.atom import Atom
from emmet.xtal.core import (
    AtomProperties,
    CellVector3D,
    NonPeriodicConfig,
    PeriodicConfig,
)

if TYPE_CHECKING:
    import os
    from typing import Any


def _parse_xyz(
    file_name: str | os.PathLike[str], extended: bool = False, trajectory: bool = False
) -> dict[str, Any] | list[dict[str, Any]]:
    """Parse a variety of xyz files as a stream.

    This class streams a file into memory to allow for parsing
    large xyz trajectory files easily.

    Parameters
    -----------
    file_name : str or PathLike
        The name of the file to stream.
    extended : bool = False
        Whether to use ASE-style extended xyz files.
        Allows for storing lattice vectors, pbc, and other
        properties on the xyz comment line.
    trajectory : bool = False
        Whether to parse multiple atomic configurations as
        a trajectory.

    Returns
    -----------
    If trajectory, a list of dict of str, otherwise a single dict.

    The minimal required keys of the dict are "species", containing
    the atomic names, and "pos", containing the Cartesian coordinates.
    """

    props = {
        "properties": [
            {"name": "species", "type": str, "rank": 1},
            {"name": "pos", "type": float, "rank": 3},
        ]
    }
    str_to_type = {"S": str, "R": float, "I": int}

    ixyz = 0
    if trajectory:
        atoms = []

    with open(file_name, "r") as xyz_file:
        for line in xyz_file:
            if ixyz == 0:
                natom = int(line.strip())

            elif ixyz == 1 and extended:
                props = {
                    v[0].lower(): v[2] if v[2] else v[1]
                    for v in re.findall(
                        r'\b(\w+)=("([^"]*)"|[^\s]+)', line  # thanks ChatGPTonk!
                    )
                }
                if "pbc" in props:
                    props["pbc"] = [
                        True if x.lower().startswith("t") else False
                        for x in props["pbc"].split()
                    ]
                if "lattice" in props:
                    props["lattice"] = np.reshape(
                        [float(x) for x in props["lattice"].split()], (3, 3)
                    )
                    if not props.get("pbc"):
                        props["pbc"] = [True, True, True]
                if "properties" in props:
                    extxyz_props = props["properties"].split(":")
                    props["properties"] = [
                        {
                            "name": extxyz_props[3 * idx],
                            "type": str_to_type.get(extxyz_props[3 * idx + 1]),
                            "rank": int(extxyz_props[3 * idx + 2]),
                        }
                        for idx in range(len(extxyz_props) // 3)
                    ]
            elif ixyz > 1 and ixyz < natom + 2:
                if ixyz == 2:
                    ions = {
                        prop["name"]: [None for _ in range(natom)]
                        for prop in props["properties"]
                    }
                vals = line.strip().split()
                i = 0
                for prop in props["properties"]:
                    if prop["rank"] == 1:
                        ions[prop["name"]][ixyz - 2] = prop["type"](vals[i])
                    else:
                        ions[prop["name"]][ixyz - 2] = [
                            prop["type"](vals[i + j]) for j in range(prop["rank"])
                        ]
                    i += prop["rank"]

                if ixyz == natom + 1:
                    for k, v in props.items():
                        if k != "properties":
                            ions[k] = v

                    if trajectory:
                        ixyz = -1
                        atoms.append(ions)
                    else:
                        break

            ixyz += 1
    if trajectory:
        return atoms
    return ions


class XyzParser(BaseModel):

    extended: bool = False
    trajectory: bool = False

    @staticmethod
    def _atom_config_from_xyz_dict(
        dct: dict[str, Any],
    ) -> NonPeriodicConfig | PeriodicConfig:
        natom = len(dct["species"])
        nulled = [None] * natom

        atoms = [Atom(name=ele) for ele in dct["species"]]
        atom_properties = [
            AtomProperties(
                **{k: dct.get(k, nulled)[i] for k in AtomProperties.model_fields}
            )
            for i in range(natom)
        ]

        cv = None
        if (cvm := dct.get("lattice")) is not None:
            cv = CellVector3D(matrix=cvm)
            return PeriodicConfig(
                atoms=atoms,
                coords=dct["pos"],
                cell=cv,
                pbc=dct.get("pbc", None),
                atom_properties=atom_properties,
            )
        return NonPeriodicConfig(
            atoms=atoms,
            coords=dct["pos"],
            atom_properties=atom_properties,
        )

    def parse_xyz(
        self,
        file_name: str,
    ) -> NonPeriodicConfig | PeriodicConfig | list[NonPeriodicConfig | PeriodicConfig]:

        parsed = _parse_xyz(
            file_name, extended=self.extended, trajectory=self.trajectory
        )

        if self.trajectory:
            return [self._atom_config_from_xyz_dict(dct) for dct in parsed]
        return self._atom_config_from_xyz_dict(parsed)
