from tvm._ffi import register_object as _register_object

from tvm.runtime import Object

from tvm.tir.expr import Var
from tvm.tir.buffer import Buffer


from . import _ffi_api
from .base import TIRFrame


@_register_object("script.builder.tir.PrimFuncFrame")
class PrimFunc(TIRFrame):
    def __init__(self, name) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.PrimFuncFrame,
            name
        )

    def __exit__(self, ptype, value, trace) -> None:
        _ffi_api.ExitPrimFuncFrame(self)


def Arg(arg):
    if isinstance(arg, Var):
        _ffi_api.ArgVar(arg)
    elif isinstance(arg, Buffer):
        _ffi_api.ArgBuffer(arg)
    else:
        assert False
    
