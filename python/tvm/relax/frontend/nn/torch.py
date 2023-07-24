# pylint: disable=missing-docstring
import inspect
from typing import Any, Callable, List

import torch

from tvm.ir import Array
from tvm.runtime import Device, NDArray, ShapeTuple, ndarray
from tvm.runtime.relax_vm import VirtualMachine

from . import core
from . import spec as _spec


class TorchModule:  # pylint: disable=too-few-public-methods
    spec: _spec.ModuleSpec
    virtual_machine: VirtualMachine
    params: List[NDArray]
    effects: List[Any]

    def __init__(
        self,
        spec: _spec.ModuleSpec,
        virtual_machine: VirtualMachine,
        params: List[NDArray],
    ):
        effects = virtual_machine["_initialize_effect"]()
        self.spec = spec
        self.virtual_machine = virtual_machine
        self.params = params
        self.effects = effects

    def __getitem__(self, method_name: str) -> Callable:
        def _find_method(method_name):
            for key, value in zip(self.spec.method_names, self.spec.method_specs):
                if method_name == key:
                    return value
            raise ValueError(f"Method `{method_name}` is not found in the module spec. {self.spec}")

        method_spec = _find_method(method_name)
        method = self.virtual_machine[method_name]

        def _closure(*args):
            if len(args) != len(method_spec.arg_names):
                raise TypeError(
                    f"Argument length mismatch. Expected {len(method_spec.args)} arguments, "
                    f"but got {len(args)} arguments. The spec is: {method_spec}"
                )
            args = [
                _torch_to_tvm(arg_name, arg_spec, arg)
                for arg_name, arg_spec, arg in zip(
                    method_spec.arg_names, method_spec.arg_specs, args
                )
            ]
            outputs, self.effects = method(*args, *self.params, *self.effects)
            return _tvm_to_torch(outputs)

        _closure.__name__ = method_name
        return _closure


@staticmethod
def _tvm_to_torch(arg):
    if isinstance(arg, (list, tuple, Array)):
        return [_tvm_to_torch(i) for i in arg]
    if isinstance(arg, ndarray.NDArray):
        return torch.utils.dlpack.from_dlpack(arg)
    if isinstance(arg, ShapeTuple):
        return list(arg)
    raise TypeError(f"Unsupported argument type: {type(arg)}")


def _torch_to_tvm(arg_name, arg_spec, arg_torch):
    if isinstance(arg_spec, _spec.Tensor):
        if not isinstance(arg_torch, torch.Tensor):
            raise TypeError(
                f"Expected argument `{arg_name}` to be `torch.Tensor`, "
                f"but got {type(arg_torch)}"
            )
        try:
            return ndarray.from_dlpack(arg_torch)
        except RuntimeError:
            device_type = arg_torch.device.type
            device_id = arg_torch.device.index or 0
            return ndarray.array(
                arg_torch.numpy(),
                device=Device(
                    Device.STR2MASK[device_type],
                    device_id,
                ),
            )
    if isinstance(arg_spec, _spec.Int):
        if not isinstance(arg_torch, int):
            raise TypeError(
                f"Expected argument `{arg_name}` to be `int`, but got {type(arg_torch)}"
            )
        return ShapeTuple([arg_torch])
    raise TypeError(f"Unsupported spec item type: {type(arg_spec)}")


def _method_spec_from_torch(
    module: core.Module,
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
