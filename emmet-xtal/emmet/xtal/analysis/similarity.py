"""Tools for matching structures."""

from __future__ import annotations

from itertools import permutations
from pydantic import BaseModel, Field
import numpy as np
from typing import TYPE_CHECKING

from emmet.xtal import SETTINGS

if TYPE_CHECKING:
    from emmet.xtal.atom import Atom
    from emmet.xtal.core import Composition, NonPeriodicConfig


class NonPeriodicMatcher(BaseModel):

    dist_tol: float = Field(SETTINGS.SYMPREC, description="Linear distance tolerance.")
    ignored_properties: list[str] | None = Field(
        None, description="Atomic properties to ignore while checking site positions."
    )

    @staticmethod
    def l2(arr1, arr2: np.ndarray) -> np.ndarray:
        disp = arr1 - arr2
        return np.einsum("ij,ij->i", disp, disp)

    def opt_coord_groups(
        self,
        g1: dict[Atom | Composition, np.ndarray],
        g2: dict[Atom | Composition, np.ndarray],
    ) -> tuple[
        dict[Atom | Composition, np.ndarray], dict[Atom | Composition, np.ndarray]
    ]:
        """ "Attempt to align grouped coordinates by minimizing distance between them."""

        idxs = list(g1)
        new_g1 = {}
        for k in idxs:
            min_dist = np.inf
            opt_perm = None
            for perm in permutations(range(len(g1[k]))):
                if (res := self.l2(g1[k][[perm]], g2[k]).mean()) < min_dist:
                    min_dist = res
                    opt_perm = perm

                # If RMS distance below tolerance, stop permuting
                if res ** (0.5) < self.dist_tol:
                    break
            new_g1[k] = g1[k][[opt_perm]]

        return new_g1, g2

    def fit(self, m1: NonPeriodicConfig, m2: NonPeriodicConfig) -> bool:

        if m1.formula != m2.formula:
            return False

        groups = [m._group_coords(ignore=self.ignored_properties) for m in (m1, m2)]

        if any(len(groups[1].get(k, [])) != len(v) for k, v in groups[0].items()):
            return False

        groups = self.opt_coord_groups(groups)
