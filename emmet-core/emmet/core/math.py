Vector3D = tuple[float, float, float]
Vector3D.__doc__ = "Real space vector"  # type: ignore

Matrix3D = tuple[Vector3D, Vector3D, Vector3D]
Matrix3D.__doc__ = "Real space Matrix"  # type: ignore

Vector6D = tuple[float, float, float, float, float, float]
Vector6D.__doc__ = "6D Voigt matrix component"  # type: ignore

MatrixVoigt = tuple[Vector6D, Vector6D, Vector6D, Vector6D, Vector6D, Vector6D]
Vector6D.__doc__ = "Voigt representation of a 3x3x3x3 tensor"  # type: ignore

Tensor3R = list[list[list[float]]]
Tensor3R.__doc__ = "Generic tensor of rank 3"  # type: ignore

Tensor4R = list[list[list[list[float]]]]
Tensor4R.__doc__ = "Generic tensor of rank 4"  # type: ignore

listVector3D = list[float]
listVector3D.__doc__ = "Real space vector as list"  # type: ignore

listMatrix3D = list[listVector3D]
listMatrix3D.__doc__ = "Real space Matrix as list"  # type: ignore
