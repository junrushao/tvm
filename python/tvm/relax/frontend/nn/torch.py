# pylint: disable=missing-docstring
import inspect
from typing import Any, Callable, List

import torch

from tvm import tir
from tvm.ir import Array, IRModule
from tvm.runtime import Device, ShapeTuple, ndarray
from tvm.runtime.relax_vm import VirtualMachine
from tvm.target import Target

from ... import pipeline, vm_build
from . import spec as _spec
from .core import Module, Tensor, _tensor_placeholder

# class TorchModule:  # pylint: disable=too-few-public-methods
#     spec: _spec.Module
#     virtual_machine: VirtualMachine
#     effects: Any
#
#     def __init__(
#         self,
#         spec: _spec.Module,
#         mod: IRModule,
#         target: Target,
#         device: Device,
#     ):
#         mod.show(black_format=False)
#         mod = pipeline.get_pipeline("zero")(mod)  # pylint: disable=no-value-for-parameter
#         self.spec = spec
#         self.virtual_machine = VirtualMachine(vm_build.build(mod, target), device)
#         self.effects = self.virtual_machine["_initialize_effect"]()
#
#     @staticmethod
#     def _convert_from_torch(arg_spec, arg):
#         if isinstance(arg_spec, Tensor):
#             if not isinstance(arg, torch.Tensor):
#                 raise TypeError(
#                     f"Expected argument `{arg_spec._expr.name_hint}` to be `torch.Tensor`, "  # pylint: disable=protected-access
#                     f"but got {type(arg)}"
#                 )
#             return ndarray.array(arg.detach().cpu().numpy())
#         if isinstance(arg_spec, tir.Var):
#             if not isinstance(arg, int):
#                 raise TypeError(
#                     f"Expected argument `{arg_spec.name}` to be `int`, " f"but got {type(arg)}"
#                 )
#             return ShapeTuple([arg])
#         raise TypeError(f"Unsupported spec item type: {type(arg_spec)}")
#
#     @staticmethod
#     def _convert_to_torch(arg):
#         if isinstance(arg, (list, tuple, Array)):
#             return [TorchModule._convert_to_torch(i) for i in arg]
#         if isinstance(arg, ndarray.NDArray):
#             return torch.from_numpy(arg.numpy())
#         if isinstance(arg, ShapeTuple):
#             return list(arg)
#         raise TypeError(f"Unsupported argument type: {type(arg)}")
#
#     def __getitem__(self, name: str) -> Callable:
#         if name not in self.spec.methods:
#             raise IndexError(f"Method `{name}` not found in compiled module. {self.spec}")
#         spec = self.spec.methods[name]
#
#         def _closure(*args):
#             if len(args) != len(spec.args):
#                 raise TypeError(
#                     f"Argument length mismatch. "
#                     f"Expected {len(spec.args)} arguments, "
#                     f"but got {len(args)} arguments. "
#                     f"The spec is: {spec}"
#                 )
#             args = [
#                 TorchModule._convert_from_torch(arg_spec, arg)
#                 for arg_spec, arg in zip(spec.args, args)
#             ]
#             outputs, self.effects = self.virtual_machine[name](*args, *self.effects)
#             return TorchModule._convert_to_torch(outputs)
#
#         _closure.__name__ = name
#         return _closure


def _method_spec_from_torch(
    module: Module,
    method_name: str,
    args_torch: List[Any],
):
    def _as_spec(arg_torch):
        if isinstance(arg_torch, torch.Tensor):
            _, dtype = str(arg_torch.dtype).rsplit(".", maxsplit=1)
            return _spec.Tensor(shape=list(arg_torch.shape), dtype=dtype)
        if isinstance(arg_torch, int):
            return _spec.Int()
        raise TypeError(f"Unsupported argument type: {type(arg_torch)}")

    method = getattr(module, method_name)
    arg_names = list(inspect.signature(method).parameters.keys())
    if len(arg_names) != len(args_torch):
        raise TypeError(f"Expected {len(arg_names)} arguments, but got {len(args_torch)} arguments")
    arg_specs = [_as_spec(i) for i in args_torch]
    return _spec.MethodSpec(method, arg_names, arg_specs)
