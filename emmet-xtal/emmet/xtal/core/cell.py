from __future__ import annotations

from functools import cached_property
import numpy as np

class Cell(np.ndarray):

    def __new__(cls, data, **kwargs):
        arr = np.asarray(data, dtype=float)
        return arr.view(cls)

    def __array_finalize__(self, obj):
        # If shape/dtype are wrong, demote to plain ndarray instead of raising
        if self.shape != (3, 3) or self.dtype != np.float64:
            # Can't mutate self's type in-place, so we flag it for __array_ufunc__
            self._invalid = True

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Delegate to plain ndarray, then re-wrap only if result is still 3x3 float
        plain_inputs = [np.asarray(x) for x in inputs]
        result = getattr(ufunc, method)(*plain_inputs, **kwargs)
        if isinstance(result, np.ndarray) and result.shape == (3, 3) and result.dtype == np.float64:
            return result.view(Matrix3x3)
        return result  # return plain ndarray if result doesn't qualify

    @cached_property
    def volume(self) -> float:
        return abs(np.linalg.det(self))

    @cached_property
    def _reciprocal(self) -> Cell:
        return Cell(
            np.array(
                [np.cross(self[(i + 1) % 3], self[(i + 2) % 3]) for i in range(3)]
            )/self.volume
        )

    @property
    def reciprocal(self) -> Cell:
        return 2*np.pi*self._reciprocal

    @cached_property
    def _vector_norms(self) -> np.ndarray:
        return np.linalg.norm(self,axis=1)

    @cached_property
    def _angles(self) -> np.ndarray:
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