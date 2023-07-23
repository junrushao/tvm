# pylint: disable=missing-docstring,too-many-lines
import warnings
from collections import OrderedDict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from tvm import tir
from tvm.ir import IRModule
from tvm.runtime import Device, NDArray, ndarray
from tvm.target import Target

from ... import expr as rx
from ...block_builder import BlockBuilder
from ...struct_info import ShapeStructInfo, TensorStructInfo
from ._tensor_op import _TensorOp

if TYPE_CHECKING:
    from . import spec as _spec
    from . import torch

IntExpr = Union[int, tir.PrimExpr]


_DEFAULT_DTYPE = "float32"


def get_default_dtype() -> str:
    return _DEFAULT_DTYPE


def set_default_dtype(dtype: str) -> None:
    global _DEFAULT_DTYPE  # pylint: disable=global-statement
    _DEFAULT_DTYPE = dtype


class Tensor(_TensorOp):
    _expr: rx.Expr

    def __init__(self, *, _expr: rx.Expr) -> None:
        def _check_tensor(expr: rx.Expr) -> None:
            assert expr.struct_info_ is not None
            assert isinstance(expr.struct_info, TensorStructInfo)
            assert expr.struct_info.ndim != -1
            assert expr.struct_info.shape is not None
            assert expr.struct_info.shape.struct_info_ is not None
            assert isinstance(expr.struct_info.shape.struct_info, ShapeStructInfo)
            assert expr.struct_info.shape.struct_info.values is not None

        _check_tensor(_expr)
        self._expr = _expr

    @staticmethod
    def from_const(data) -> "Tensor":
        return Tensor(_expr=rx.const(data))

    @staticmethod
    def from_scalar(data: Union[int, float], dtype: str) -> "Tensor":
        return Tensor(_expr=rx.const(data, dtype=dtype))

    @property
    def shape(self) -> List[IntExpr]:
        def _simplify(expr: tir.PrimExpr):
            return expr.value if isinstance(expr, tir.IntImm) else expr

        shape_sinfo: ShapeStructInfo = self._expr.struct_info.shape.struct_info
        return [_simplify(x) for x in shape_sinfo.values]

    @property
    def ndim(self) -> int:
        return self._expr.struct_info.ndim

    @property
    def dtype(self) -> str:
        return self._expr.struct_info.dtype

    def to(self, dtype: Optional[str] = None) -> None:  # pylint: disable=invalid-name
        # pylint: disable=protected-access
        assert isinstance(self._expr, rx.Var)
        if dtype is not None:
            self._expr = _tensor_placeholder("p", self.shape, dtype=dtype)._expr
        # pylint: enable=protected-access

    def __repr__(self) -> str:
        return f'Tensor({self.shape}, "{self.dtype}")'


class Parameter(Tensor):
    _data: Optional[NDArray]

    def __init__(self, shape: Sequence[IntExpr], dtype: Optional[str] = None) -> None:
        if dtype is None:
            dtype = get_default_dtype()
        super().__init__(_expr=_tensor_placeholder("p", shape, dtype=dtype)._expr)
        self._data = None

    @property
    def data(self) -> Optional[NDArray]:
        return self._data

    @data.setter
    def data(
        self,
        data: Union[
            None,
            NDArray,
            np.ndarray,
            "torch.Tensor",
        ],
    ) -> None:
        if data is None:
            self._data = data
            return
        # Try to do zero-copy if possible
        if isinstance(data, NDArray):
            pass
        elif isinstance(data, np.ndarray):
            data = ndarray.array(data)
        elif hasattr(data, "__dlpack__"):
            data = ndarray.from_dlpack(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        if data.shape != tuple(self.shape):
            raise ValueError(f"Shape mismatch: expected {tuple(self.shape)}, got {data.shape}")
        if data.dtype != self.dtype:
            raise ValueError(f"Dtype mismatch: expected {self.dtype}, got {data.dtype}")
        self._data = data

    def to(self, dtype: Optional[str] = None) -> None:  # pylint: disable=invalid-name
        if dtype is not None and self._data is not None:
            raise ValueError(
                "Changing the dtype of a Parameter that has been bound to concrete "
                "data is not recommended. It might lead to potential precision loss "
                "or other unexpected behaviors"
            )
        super().to(dtype=dtype)


class Effect:
    def emit_init(self, name_hint: str, builder: BlockBuilder) -> List[rx.DataflowVar]:
        raise NotImplementedError

    def create(self, name_hint: str) -> List[rx.Var]:
        raise NotImplementedError

    def finalize(self) -> List[rx.Var]:
        raise NotImplementedError

    def to(self, dtype: Optional[str] = None) -> None:  # pylint: disable=invalid-name
        pass  # do nothing by default because Effect is not necessarily a Tensor


class Module:
    def named_parameters(
        self,
        prefix: str = "",
    ) -> Iterator[Tuple[str, Parameter]]:
        yield from _attribute_finder(
            self,
            prefix,
            condition_yield=lambda x: isinstance(x, Parameter),
        )

    def state_dict(
        self,
        *,
        prefix: str = "",
        destination: Optional[Dict[str, Parameter]] = None,
    ) -> Dict[str, Parameter]:
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters(prefix=prefix):
            destination[name] = param
        return destination

    def load_state_dict(
        self,
        state_dict: Dict[str, Parameter],
        strict: bool = True,
    ) -> Tuple[List[str], List[str]]:  # TODO(@junrushao): switch to NamedTuple
        self_state_dict = self.state_dict()
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        for key, value in state_dict.items():
            if key not in self_state_dict:
                unexpected_keys.append(key)
                continue
            if value.data is None:
                raise ValueError(f"Parameter {key} is not set to any concrete tensor")
            assert isinstance(value.data, NDArray)  # TODO(@junrushao): expand to more types
            self_state_dict.pop(key).set_data(value.data)
        missing_keys = list(self_state_dict.keys())
        if strict and (missing_keys or unexpected_keys):
            raise KeyError(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
        return missing_keys, unexpected_keys

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self, "forward"):
            raise NotImplementedError(f"Module {type(self)} does not have a `forward` method")
        return self.forward(*args, **kwargs)  # pylint: disable=no-member

    def tensor_expr_op(
        self,
        tensor_expr_func: Callable,
        name_hint: str,
        args: List[Union[Tensor, tir.Var]],
        *,
        attrs: Optional[Dict[str, Any]] = None,
    ):
        def _convert(arg):
            if isinstance(arg, Tensor):
                return arg._expr  # pylint: disable=protected-access
            return arg

        from .op import _wrap_nested  # pylint: disable=import-outside-toplevel

        return _wrap_nested(
            BlockBuilder.current().emit_te(
                tensor_expr_func,
                *[_convert(arg) for arg in args],
                primfunc_name_hint=name_hint,
                primfunc_attrs=attrs,
            ),
            name=name_hint,
        )

    def to(self, dtype: Optional[str] = None) -> None:  # pylint: disable=invalid-name
        for _, item in self.__dict__.items():
            if hasattr(item, "to") and callable(item.to):
                item.to(dtype=dtype)

    def export_tvm(self, spec: "_spec.Module") -> IRModule:
        from . import spec as _spec  # pylint: disable=import-outside-toplevel

        spec = _spec.ModuleSpec.from_raw(spec, self)
        mod, params = _spec.SpecBuilder().build(spec)
        return mod, params

    def jit(
        self,
        spec: "_spec.Module",
        target: Union[str, Target] = "llvm",
        device: str = "cpu",
        out_format: str = "torch",
    ) -> Callable:
        from . import spec as _spec  # pylint: disable=import-outside-toplevel

        spec = _spec.ModuleSpec.from_raw(spec, self)
        mod, params = _spec.SpecBuilder().build(spec)

        if out_format == "torch":
            from . import torch  # pylint: disable=import-outside-toplevel

            return torch.TorchModule(
                spec=spec,
                mod=self.export_tvm(spec),
                target=target,
                device=_str_to_device(device),
            )
        raise ValueError(f"Unknown out_format: {out_format}")


class ModuleList(Module):
    def __init__(self, modules: List[Module]):
        self.modules = modules

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, idx):
        return self.modules[idx]

    def __setitem__(self, idx, module):
        self.modules[idx] = module

    def __len__(self):
        return len(self.modules)

    def to(self, dtype: Optional[str] = None) -> None:  # pylint: disable=invalid-name
        for module in self.modules:
            module.to(dtype=dtype)

    def forward(self, x):  # pylint: disable=invalid-name
        for module in self.modules:
            x = module(x)
        return x


def _attribute_finder(
    root: Any,
    prefix: str,
    condition_yield: Callable[[Any], bool],
):
    for name, item in root.__dict__.items():
        if condition_yield(item):
            yield prefix + name, item
        elif isinstance(item, ModuleList):
            for i, subitem in enumerate(item):
                yield from _attribute_finder(
                    subitem,
                    prefix + name + f".{i}.",
                    condition_yield,
                )
        elif isinstance(item, Module):
            yield from _attribute_finder(
                item,
                prefix + name + ".",
                condition_yield,
            )


def _str_to_device(device: str) -> Device:
    # TODO: upstream it
    split = device.split(":")
    if len(split) > 2:
        raise ValueError(f"Invalid device: {device}")
    device_type = split[0]
    if device_type not in Device.STR2MASK:
        raise ValueError(f"Unsupported device type: {device_type}")
    if len(split) == 1:
        device_id = 0
    else:
        device_id = int(split[1])
    return Device(Device.STR2MASK[device_type], device_id)


def _tensor_placeholder(
    name: str,
    shape: Sequence[IntExpr],
    dtype: str,
) -> "Tensor":
    new_shape = []
    for expr in shape:
        if isinstance(expr, (int, tir.IntImm)):
            expr = int(expr)
            assert expr >= 0
            new_shape.append(expr)
            continue
        if not isinstance(expr, tir.PrimExpr):
            raise TypeError(f"Invalid shape: {shape}")
        assert expr.dtype == "int64"
        new_shape.append(expr)
    return Tensor(
        _expr=rx.Var(
            name_hint=name,
            struct_info=TensorStructInfo(
                shape=new_shape,
                dtype=dtype,
            ),
        )
    )
