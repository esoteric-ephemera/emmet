"""Tools for analyzing trajectories."""

from __future__ import annotations

from functools import cached_property
import logging
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from emmet.xtal.base import Atom, AtomSymbol

if TYPE_CHECKING:
    from emmet.xtal.core import NonPeriodicConfig, PeriodicConfig

logger = logging.getLogger(__name__)


class DiffusionCoeff(BaseModel):
    """Lightweight model of extrapolated diffusion coefficient data.

    NOTE: `d_units` specifies the length units used in
    `msd`, `d`, and `intercept`, as well as the time units
    used in `t` and `d`.


    Attributes
    -----------
    t : list of float
        The times used to fit the diffusion coefficient
    msd : list of float
        The mean squared displacements used in the fit.
    d : float or None
        The diffusion coefficient, if fitted.
    intercept : float or None
        The optional intercept of a linear fit.
    d_units : str or None
        The units of the diffusion coefficient.
    """

    t: list[float]
    msd: list[float]

    d: float | None = None
    d_units: str | None = None
    intercept: float | None = None


class TrajectoryAnalyzer:
    """Analyze a trajectory.

    Parameters
    -----------
    elements : list of int
        The atomic numbers of each coordinate
    cart_coords : numpy array
        Array of cartesian coordinates over the trajectory.
        This should be a (num ionic steps) x (num atoms) x 3 array
    lattice : numpy array or None
        For PBC, either a single 3x3 or (num ionic steps) x (3 x 3)
        numpy array. In the former case, the lattice is assumed to
        be constant.

        For molecules, this should be None.
    """

    __slots__ = ["atomic_numbers", "element_groups", "cart_coords", "lattice"]

    def __init__(
        self,
        atomic_numbers: list[int],
        cart_coords: np.ndarray,
        lattice: np.ndarray | None = None,
    ) -> None:

        self.atomic_numbers = atomic_numbers
        self.element_groups: dict[str, list[int]] = {}
        for i, atomic_num in enumerate(self.atomic_numbers):
            ele = AtomSymbol.from_atomic_number(atomic_num)
            if ele not in self.element_groups:
                self.element_groups[ele] = []
            self.element_groups[ele].append(i)

        self.cart_coords = cart_coords
        self.lattice = lattice

    @property
    def num_steps(self) -> int:
        """The number of trajectory steps."""
        return self.cart_coords.shape[0]

    @classmethod
    def from_frames(
        cls,
        configs: list[NonPeriodicConfig | PeriodicConfig],
        constant_cell: bool = False,
        unfold: bool = False,
    ):
        """Begin analysis from Atom/Trajectory.

        Parameters
        -----------
        traj : List of either **only** NonPeriodicConfig or of only PeriodicConfig.
        constant_cell : bool = False
            For PeriodicConfig, whether the cell vectors were held constant.
        unfold : bool = False
            NOT IMPLEMENTED YET: whether to attempt to unfold the
            trajectory along PBC if a lattice is specified
        """

        if not all(isinstance(c, type(configs[0])) for c in configs[1:]):
            raise ValueError(
                "Cannot accommodate mixed trajectory types; your "
                f"first frame was {type(configs[0])}, the rest differed."
            )

        lattice = None
        if isinstance(configs[0], PeriodicConfig):
            if constant_cell:
                lattice = np.array(configs[0].cell.matrix)
            else:
                lattice = np.array([c.cell.matrix for c in configs])

        # if unfold:
        #     # TODO check for umklapp scattering
        #     cart_coords = traj.cart_coords
        # else:
        #     cart_coords = traj.cart_coords

        return cls(
            atomic_numbers=configs[0].atomic_numbers,
            cart_coords=np.array([c.coords for c in configs]),
            lattice=lattice,
        )

    @cached_property
    def com(self) -> np.ndarray:
        """Obtain the center of mass (COM) as a function of time."""
        masses = np.array(
            [
                Atom.from_atomic_number(atomic_num).mass_amu
                for atomic_num in self.atomic_numbers
            ]
        )
        weighted = np.einsum("k,ikj->ikj", masses, self.cart_coords) / masses.sum()
        return np.sum(weighted, axis=1)

    def msd(self, elements: list[str] | None = None) -> dict[str, np.ndarray]:
        """Obtain the mean-squared displacement (MSD).

        Parameters
        -----------
        elements : list of str or None
            If a list of str, specifies the elements to
            compute the MSD for.
            If None, computes it for all elements in `element_groups`

        Returns
        -----------
        Dict of element name to a numpy array containing
        the MSD averaged over the square displacements of
        those elements.
        """
        elements = elements or list(self.element_groups)
        msd: dict[str, np.ndarray] = {}
        drift = (self.com - self.com[0])[:, np.newaxis]
        disp = self.cart_coords - self.cart_coords[0] - drift
        for ele in elements:
            ele_idx = self.element_groups[ele]
            msd[ele] = np.einsum(
                "ikj,ikj->i", disp[:, ele_idx, :], disp[:, ele_idx, :]
            ) / len(ele_idx)
        return msd

    def rmsd(self, elements: list[str] | None = None) -> dict[str, np.ndarray]:
        """Get the root mean-squared displacements (RMSD)."""
        return {ele: msd ** (0.5) for ele, msd in self.msd(elements=elements).items()}

    @cached_property
    def rmsf(self) -> dict[str, np.ndarray]:
        """Get the root mean-squared fluctuation (RMSF)."""
        rmsf = {}
        avg_pos = np.einsum("ijk->jk", self.cart_coords) / self.cart_coords.shape[0]
        drift = (self.com - self.com[0])[:, np.newaxis]
        disp = self.cart_coords - avg_pos - drift
        for ele, ele_idx in self.element_groups.items():
            rmsf[ele] = (
                np.einsum("ikj,ikj->i", disp[:, ele_idx, :], disp[:, ele_idx, :])
                / len(ele_idx)
            ) ** (0.5)
        return rmsf

    def diffusion_coefficient(
        self,
        traj_window: tuple[int | None, int | None] = (None, None),
        elements: list[str | AtomSymbol] | None = None,
        distance_unit: str | None = None,
        time_step: str | float | None = None,
        convert_to: str | None = None,
        verbose: bool = False,
    ) -> dict[str, DiffusionCoeff]:
        """Perform a linear least-squares fit for the diffusion coefficient.

        For automatic unit conversion, `distance_unit`, `time_step`, and
        `convert_to` must be specified.
        Unit conversion is performed with `pint`.

        Parameters
        -----------
        traj_window : 2-tuple of int or None
            Specifies the lower and upper bounds of the trajectory
            to use in fitting.
            If an index is None, the corresponding extreme index will
            be taken, either 0 or `self.num_steps`
        elements : list of str or AtomSymbol, or None
            If a list of str or AtomSymbol, specifies the elements to
            compute the MSD for.
            If None, computes it for all elements in `element_groups`
        distance_unit : str or None
            If not None, the distance unit used in the coordinates,
            e.g., "angstrom"
        time_step : str or float or None
            If a float, the time step without units.
            If a str, the time step with units.
            If None, the time step is dimensionless.
        convert_to : str or None
            If a str, the units to convert to, e.g., "cm^2/s"
        verbose : bool = False
            If true, raises warnings when the diffusion coefficient
            fit fails or produces unphysical values.

        Returns
        -----------
        Dict of element name to DiffusionCoeff
        """

        window = slice(traj_window[0] or 0, traj_window[1] or self.num_steps)
        elements = AtomSymbol[elements] or list(self.element_groups)

        if convert_to:
            if (not distance_unit or not time_step) or (
                distance_unit and isinstance(time_step, float)
            ):
                raise ValueError(
                    "You must specify the units of distance and the time "
                    "step with corresponding units, e.g., `time_step = '1 fs'` "
                    "to use the unit conversion features."
                )

            try:
                from pint import UnitRegistry
            except ImportError:
                raise ImportError(
                    "You must `pip install pint` to use the unit "
                    "conversion features of this class."
                )
            unit_reg = UnitRegistry()
            time_step_unitized = unit_reg(time_step)  # type: ignore[arg-type]
            dt = time_step_unitized.m
            input_units = f"{distance_unit}^2/{time_step_unitized.u}"
            inp_units = unit_reg(input_units)

            unit_conv = {
                "diffusivity": inp_units.to(convert_to).m,
                "length": distance_unit,
                "time": str(time_step_unitized.u),
            }
            for unit_long_name in unit_reg(convert_to)._units:
                dimension = str(unit_reg(unit_long_name).dimensionality)[1:-1]
                unit_conv[dimension] = (
                    unit_reg(unit_conv[dimension]).to(unit_long_name).m
                )

        else:
            unit_conv = {k: 1.0 for k in ("diffusivity", "length", "time")}
            dt = time_step or 1.0

        time = dt * (1 + np.arange(self.num_steps))

        processed = {}
        msd = self.msd(elements=elements)
        for ele in elements:
            try:
                intercept, slope = _least_squares_fit(time[window], msd[ele][window])
            except Exception as exc:
                intercept = None
                slope = None
                if verbose:
                    logger.warning(str(exc))

            if slope < 0:
                if verbose:
                    logger.warning(
                        f"Zeroing negative {ele} diffusion coefficient "
                        f"{slope* unit_conv['diffusivity'] / 6:.2e}"
                    )
                slope = 0.0

            processed[ele] = DiffusionCoeff(
                t=time[window] * unit_conv["time"],
                msd=msd[ele][window] * unit_conv["length"] ** 2,
                d=slope * unit_conv["diffusivity"] / 6 if slope is not None else None,
                intercept=(
                    intercept * unit_conv["length"] ** 2
                    if intercept is not None
                    else None
                ),
                d_units=convert_to,
            )

        return processed


def _least_squares_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design_matrix = np.vstack([np.ones_like(x), x]).T
    return np.linalg.lstsq(design_matrix, y, rcond=None)[0]
