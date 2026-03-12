"""Tools for matching structures."""

from __future__ import annotations

from itertools import permutations, product
from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation
import numpy as np
from typing import TYPE_CHECKING

import amd

from emmet.xtal import SETTINGS
from emmet.xtal.core import Molecule, Material


if TYPE_CHECKING:
    from collections.abc import Generator


class MoleculeMatcher(BaseModel):
    """Match non-periodic collections of atoms, e.g., atoms, molecules and clusters."""

    tolerance: float = Field(
        SETTINGS.LTOL,
        description="The linear distance tolerance in the RMSD between two molecules for gauging structural match.",
    )

    @staticmethod
    def centroid(v: np.ndarray) -> np.ndarray:
        return np.mean(v, axis=0)

    def kabsch(
        self, v0: np.ndarray, v1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        if v0.shape != v1.shape:
            raise ValueError(f"Unequal vector shapes: {v0.shape} vs. {v1.shape}")

        v0 = np.array(v0)
        v1 = np.array(v1)
        centroids = (self.centroid(v0), self.centroid(v1))
        translation = centroids[0] - centroids[1]
        rotation, rssd = Rotation.align_vectors(
            v0 - centroids[0],
            v1 - centroids[1],
        )
        return translation, rotation, rssd / v0.shape[0] ** (0.5)

    def reorder_sites(self, m: Molecule):
        centroid = self.centroid(m.coords)
        sorted_idx = sorted(
            range(m.num_sites),
            key=lambda x: (
                m[x].species.elements[0].Z,
                np.linalg.norm(m[x].coords - centroid),
            ),
        )
        return self.permute_molecule(m, sorted_idx)

    @staticmethod
    def permute(m: Molecule, permutation: list[int]):
        return Molecule(
            [m.atomic_numbers[idx] for idx in permutation],
            [m.coords[idx] for idx in permutation],
        )

    def generate_permutations(
        self, m: Molecule
    ) -> Generator[tuple[tuple[int, ...], ...]]:
        sorted_m = self.reorder_sites(m)
        ranges: dict[int, tuple[int, int] | None] = {
            ele: None for ele in sorted_m.composition
        }
        for idx in range(sorted_m.num_sites):
            if idx == 0:
                start = idx
                last_ele = sorted_m.atomic_numbers[idx]
            if (curr_ele := sorted_m.atomic_numbers[idx]) != last_ele:
                ranges[last_ele] = (start, idx)
                start = idx
                last_ele = curr_ele

        ranges[last_ele] = (start, idx + 1)
        sorted_ele = sorted(ranges, key=lambda x: x.atomic_number)

        return product(*[permutations(range(*ranges[ele])) for ele in sorted_ele])  # type: ignore[misc]

    def fit(self, m0: Molecule, m1: Molecule):
        if m0.formula != m1.formula:
            return False

        m0_sorted_coords = self.reorder_sites(m0).coords
        for _perm in self.generate_permutations(m1):
            idx = []
            for idx_set in _perm:
                idx.extend(list(idx_set))

            _, _, rmsd = self.kabsch(m0_sorted_coords, self.permute(m1, idx).coords)

            if rmsd < self.tol:
                return True
        return False


class MaterialMatcher(BaseModel):
    """Match periodic configurations of atoms."""

    k_per_max_num_sites : float = Field(125, description="The k parameter in the average minimum distance model.")
    tolerance : float = Field(0.8, description="The distance tolerance for saying that two periodic configurations differ.")

    @staticmethod
    def _periodic_config_to_periodic_set(p : Material) -> amd.PeriodicSet:
        """Convert a Material to an amd.PeriodicSet."""
        return amd.PeriodicSet(
            motif = list(p.coords),
            cell = list(p.cell),
            name = "periodic_config",
            types = p.atomic_numbers,
        )

    def get_avg_min_dist(self, p0 : Material, p1 : Material) -> float:
        """Get the average minimum distance between two Material."""
        k = self.k_per_max_num_sites*max(p.num_sites for p in (p0,p1))
        return amd.EMD(
            *[
                amd.PDD(
                    self._periodic_config_to_periodic_set(ps), k
                )
                for ps in (pset1,pset2)
            ]
        )

    def fit(self, p0 : Material, p1 : Material) -> bool:
        """Check if the average minimum distance between two Material is below a tolerance."""
        return self.get_avg_min_dist(p0,p1) < self.tolerance