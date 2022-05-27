from tvm._ffi import register_object as _register_object

from tvm.runtime import Object

from . import _ffi_api
from .base import TIRFrame


@_register_object("script.builder.tir.ForFrame")
class ForFrame(TIRFrame):
    def __init__(self, name) -> None:
        pass

    def __exit__(self, ptype, value, trace) -> None:
        _ffi_api.ExitForFrame(self)


def Serial(min_val, extent, attrs):
    return _ffi_api.Serial(min_val, extent, attrs)

def Parallel(min_val, extent, attrs):
    return _ffi_api.Parallel(min_val, extent, attrs)

def Vectorized(min_val, extent, attrs):
    return _ffi_api.Vectorized(min_val, extent, attrs)

def Unroll(min_val, extent, attrs):
    return _ffi_api.Unroll(min_val, extent, attrs)

def ThreadBinding(min_val, extent, attrs):
    return _ffi_api.ThreadBinding(min_val, extent, attrs)

def Grid(min_val, extent, attrs):
    return _ffi_api.Grid(min_val, extent, attrs)
