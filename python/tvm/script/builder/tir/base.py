from tvm._ffi import register_object as _register_object

from tvm.runtime import Object
from ..frame import Frame

from . import _ffi_api

@_register_object("script.builder.tir.TIRFrame")
class TIRFrame(Frame):
    pass
