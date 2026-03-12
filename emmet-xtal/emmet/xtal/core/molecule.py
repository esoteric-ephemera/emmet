
import numpy as np

import warnings

from emmet.xtal.core.atom import AtomsMixIn


class Molecule(AtomsMixIn):

    __slots__ = ["z", "cart_coords", "charges", "spins", "velocities", "degrees_of_freedom"]

    def __init__(
        self,
        elements : list[int | str],
        cart_coords : list[list[float]],
        charges : list[float] | None = None,
        spins : list[float] | None = None,
        velocities : list[list[float]] | None = None,
        degrees_of_freedom : list[list[bool]] | None = None,
    ) -> None:
        num_sites = len(elements)
        
        self.z = np.array([Atom(ele).Z if isinstance(ele,str) else ele for ele in elements], dtype=int)
        self.cart_coords = np.array(cart_coords, dtype=float)
        if not self.cart_coords.shape == (num_sites,3):
            raise ValueError(
                f"Misaligned shapes of `elements` ({num_sites}) and cartesian coordinates ({cart_coords.shape})"
            )

        self.charges, self.spins, self.velocities, self.degrees_of_freedom = self.parse_atom_properties(
            num_sites, charges, spins, velocities, degrees_of_freedom
        )

    @classmethod
    def from_jsonable(cls, dct: dict ):
        return cls(
            *(
                dct.get(k) for k in ("elements","cart_coords",)
            ),
            **{
                k : dct.get(k)
                for k in ("charges","spins","velocities","degrees_of_freedom")
            }
        )
