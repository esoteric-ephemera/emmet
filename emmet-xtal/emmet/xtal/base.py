"""Define core schemas for xtal."""

from __future__ import annotations

from enum import Enum
from importlib import import_module
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Any


class PackageInterface(Enum):
    """Define known interchange formats."""

    PMG = "pymatgen"
    ASE = "ase"
    PANDAS = "pandas"
    ARROW = "pyarrow"

    @classmethod
    def _missing_(cls, value):
        """Mimic StrEnum behavior."""
        for member in cls:
            if member.name == value:
                return cls[value]
            elif member.value == value:
                return cls(value)


INSTALLED_PACKAGES: dict[PackageInterface, bool] = {}
for p in PackageInterface:
    try:
        import_module(p.value)
        INSTALLED_PACKAGES[p] = True
    except ModuleNotFoundError:
        INSTALLED_PACKAGES[p] = False


class XtalSettings(BaseSettings):
    """Define settings used across emmet-xtal."""

    SYMPREC: float = Field(
        default=0.01,
        description="The uncertainty in Cartesian distances used in symmetry determination.",
    )

    ANGPREC: float = Field(
        default=5,
        description="The uncertainty in angles (units of degrees) used in symmetry determination.",
    )

    LTOL: float = Field(
        default=0.5,
        description=(
            "The maxmimum RMSD in two sets of coordinates for which "
            "two non-periodic configurations are said to be the same."
        ),
    )


SETTINGS = XtalSettings()


class ShapeShifter(BaseModel):
    """Base schema for representing objects."""

    def to(self, fmt: PackageInterface | str, **kwargs) -> Any:
        """Convert current object to an external format.

        Parameters
        -----------
        fmt : PackageInterface or str equivalent
            The format to convert to.
        **kwargs :
            Any kwargs recognized by the output external object's class.
        """
        _fmt = PackageInterface(fmt)
        if INSTALLED_PACKAGES[_fmt]:
            if method := getattr(self, f"_to_{_fmt.value}", None):
                return method(**kwargs)
            else:
                raise AttributeError(
                    f"{self.__class__.__name__} does not yet have an "
                    f"interface to {_fmt.value} implemented."
                )
        else:
            raise ModuleNotFoundError(f"Please `pip install {_fmt.value}`.")

    @classmethod
    def from_obj(cls, obj: Any, fmt: PackageInterface | str, **kwargs) -> Any:
        """Intake and convert an external object.

        Parameters
        -----------
        obj : The object to convert
        fmt : PackageInterface or str equivalent
            The format to convert from.
        **kwargs :
            Any kwargs recognized by the output external object's class.
        """
        _fmt = PackageInterface(fmt)
        if INSTALLED_PACKAGES[_fmt]:
            if method := getattr(cls, f"_from_{_fmt.value}", None):
                return method(obj, **kwargs)
            else:
                raise AttributeError(
                    f"{cls.__name__} does not yet have an "
                    f"interface to {_fmt.value} implemented."
                )
        else:
            raise ModuleNotFoundError(f"Please `pip install {_fmt.value}`.")
