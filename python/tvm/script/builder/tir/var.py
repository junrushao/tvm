from tvm._ffi import register_object as _register_object

from tvm.runtime import Object

from . import _ffi_api


def Buffer(shape, dtype, name="buffer", storage_scope=""):
    return _ffi_api.Buffer(shape, dtype, name, storage_scope)