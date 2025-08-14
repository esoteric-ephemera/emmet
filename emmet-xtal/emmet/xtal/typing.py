from __future__ import annotations

from typing import TypeAlias

null_t: TypeAlias = None
"""Null type for missing package imports."""

vector_3_t: TypeAlias = tuple[float, float, float]
"""Define 3-vector type."""

matrix_3x3_t: TypeAlias = tuple[vector_3_t, vector_3_t, vector_3_t]
"""Define 3x3 matrix type."""

list_vector_3_t: TypeAlias = list[vector_3_t]
"""List of 3-vectors, such as interatomic forces."""

bool_3_t: TypeAlias = tuple[bool, bool, bool]
"""Set of three booleans, such as PBC."""
