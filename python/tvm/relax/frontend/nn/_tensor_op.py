# pylint: disable=missing-docstring,too-many-lines
from typing import Any

from tvm import tir


def _op():
    from tvm.relax.frontend.nn import op  # pylint: disable=import-outside-toplevel

    return op


def _convert_scalar(scalar, ref) -> Any:
    from .core import Tensor  # pylint: disable=import-outside-toplevel

    if isinstance(scalar, Tensor):
        return scalar
    if isinstance(scalar, (tir.FloatImm, tir.IntImm)):
        return Tensor.from_scalar(scalar.value, dtype=ref.dtype)
    if isinstance(scalar, (int, float)):
        return Tensor.from_scalar(scalar, dtype=ref.dtype)
    return scalar


class _TensorOp:
    def __add__(self, other):
        other = _convert_scalar(other, self)
        return _op().add(self, other)

    def __mul__(self, other):
        other = _convert_scalar(other, self)
        return _op().multiply(self, other)

    def __truediv__(self, other):
        other = _convert_scalar(other, self)
        return _op().divide(self, other)

    def astype(self, dtype):
        return _op().astype(self, dtype)

    def maximum(self, other):
        other = _convert_scalar(other, self)
        return _op().maximum(self, other)

    def minimum(self, other):
        other = _convert_scalar(other, self)
        return _op().minimum(self, other)

    def reshape(self, shape):
        return _op().reshape(self, shape)

    def permute_dims(self, axes):
        return _op().permute_dims(self, axes)
