# pylint: disable=missing-docstring,too-many-lines,invalid-name,protected-access
from typing import List, Optional, Sequence, Union

from tvm import tir as _tir
from tvm.relax import op as _op

from .core import IntExpr, Tensor, _wrap_nested


def add(a: Tensor, b: Tensor, name: str = "add") -> Tensor:
    return _wrap_nested(_op.add(a._expr, b._expr), name)


def multiply(a: Tensor, b: Tensor, name: str = "mul") -> Tensor:
    return _wrap_nested(_op.multiply(a._expr, b._expr), name)


def divide(a: Tensor, b: Tensor, name: str = "divide") -> Tensor:
    return _wrap_nested(_op.divide(a._expr, b._expr), name)


def matmul(a: Tensor, b: Tensor, out_dtype: Optional[str] = None, name: str = "matmul") -> Tensor:
    return _wrap_nested(_op.matmul(a._expr, b._expr, out_dtype=out_dtype), name)


def permute_dims(x: Tensor, axes: Optional[List[int]] = None, name: str = "permute_dims") -> Tensor:
    return _wrap_nested(_op.permute_dims(x._expr, axes=axes), name)


def silu(x: Tensor, name: str = "silu") -> Tensor:
    return _wrap_nested(_op.nn.silu(x._expr), name)


def take(x: Tensor, indices: Tensor, axis: Optional[int] = None, name="take") -> Tensor:
    return _wrap_nested(_op.take(x._expr, indices._expr, axis), name)


def reshape(x: Tensor, shape: Sequence[IntExpr], name="reshape") -> Tensor:
    return _wrap_nested(_op.reshape(x._expr, shape), name)


def repeat(
    x: Tensor,
    repeats: int,
    axis: Optional[int] = None,
    name="repeat",
) -> Tensor:
    return _wrap_nested(_op.repeat(x._expr, repeats, axis), name)


def rms_norm(
    x: Tensor,
    weight: Tensor,
    axes: Union[int, List[int]],
    epsilon: float = 1e-5,
    name: str = "rms_norm",
) -> Tensor:
    return _wrap_nested(_op.nn.rms_norm(x._expr, weight._expr, axes, epsilon), name)


def astype(x: Tensor, dtype: str, name: str = "astype") -> Tensor:
    return _wrap_nested(_op.astype(x._expr, dtype), name)


def maximum(x1: Tensor, x2: Tensor, name: str = "maximum"):
    return _wrap_nested(_op.maximum(x1._expr, x2._expr), name)


def minimum(x1: Tensor, x2: Tensor, name: str = "minimum"):
    return _wrap_nested(_op.minimum(x1._expr, x2._expr), name)


def softmax(x: Tensor, axis: int = -1, name: str = "softmax") -> Tensor:
    return _wrap_nested(_op.nn.softmax(x._expr, axis), name)


def squeeze(x: Tensor, axis: int = -1, name: str = "squeeze") -> Tensor:
    return _wrap_nested(_op.squeeze(x._expr, axis), name)


def broadcast_to(x: Tensor, shape: Sequence[IntExpr], name: str = "broadcast_to") -> Tensor:
    return _wrap_nested(_op.broadcast_to(x._expr, shape), name)


def triu(x: Tensor, diagonal: int = 0, name: str = "triu") -> Tensor:
    return _wrap_nested(_op.triu(x._expr, diagonal), name)


def full(
    shape: Sequence[IntExpr],
    fill_value: Tensor,
    dtype: str = "float32",
    name: str = "full",
) -> Tensor:
    from tvm import relax as rx  # pylint: disable=import-outside-toplevel

    if isinstance(fill_value, (_tir.FloatImm, _tir.IntImm)):
        fill_value = rx.const(fill_value.value, dtype=dtype)
    elif isinstance(fill_value, (int, float)):
        fill_value = rx.const(fill_value, dtype=dtype)
    else:
        fill_value = fill_value._expr
    return _wrap_nested(_op.full(shape, fill_value, dtype), name)


def zeros(
    shape: Sequence[IntExpr],
    dtype: str = "float32",
    name: str = "zeros",
) -> Tensor:
    return _wrap_nested(_op.zeros(shape, dtype), name)


def print_(array: Tensor):
    from . import spec  # pylint: disable=import-outside-toplevel

    spec.SpecBuilder.current().io_effect.print_(array)
