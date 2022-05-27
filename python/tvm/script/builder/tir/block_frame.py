from tvm._ffi import register_object as _register_object
from .base import TIRFrame

from tvm.runtime import Object

from . import _ffi_api

@_register_object("script.builder.tir.BlockFrame")
class Block(TIRFrame):
    def __init__(self, name) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.BlockFrame,
            name
        )

    def __exit__(self, ptype, value, trace) -> None:
        _ffi_api.ExitBlockFrame(self)


def push_block_var(iter_var, binding):
    return _ffi_api.PushBlockVar(iter_var, binding)

def Spatial(dom, binding, dtype):
    return _ffi_api.Spatial(dom, binding, dtype)

def Reduce(dom, binding, dtype):
    return _ffi_api.Reduce(dom, binding, dtype)

def Remap(kinds, bindings, dtype):
    return _ffi_api.Remap(kinds, bindings, dtype)