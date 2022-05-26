from tvm._ffi import register_object as _register_object

from tvm.runtime import Object

from . import _ffi_api


@_register_object("script.builder.tir.BlockFrame")
class BlockFrame(Object):
    def __init__(self, name: str) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.BlockFrame,
            name
        )

# @_register_object("script.builder.tir.Serial")
class SerialForFrame(Object):
    def __init__(self, 
        min_val,
        extent,
        attrs):
        self.__init_handle_by_constructor__(
            _ffi_api.SerialForFrame,
            min_val,
            extent,
            attrs
        )

# @_register_object("script.builder.tir.Parallel")
class ParallelForFrame(Object):
    def __init__(self, 
        min_val,
        extent,
        attrs):
        self.__init_handle_by_constructor__(
            _ffi_api.ParallelForFrame,
            min_val,
            extent,
            attrs
        )

# @_register_object("script.builder.tir.Vectorized")
class VectorizedForFrame(Object):
    def __init__(self, 
        min_val,
        extent,
        attrs):
        self.__init_handle_by_constructor__(
            _ffi_api.VectorizedForFrame,
            min_val,
            extent,
            attrs
        )

# @_register_object("script.builder.tir.Unroll")
class UnrollForFrame(Object):
    def __init__(self, 
        min_val,
        extent,
        attrs):
        self.__init_handle_by_constructor__(
            _ffi_api.UnrollForFrame,
            min_val,
            extent,
            attrs
        )

# @_register_object("script.builder.tir.ThreadBinding")
class ThreadBindingForFrame(Object):
    def __init__(self, 
        min_val,
        extent,
        thread,
        attrs):
        self.__init_handle_by_constructor__(
            _ffi_api.ThreadBindingForFrame,
            min_val,
            extent,
            thread,
            attrs
        )

# @_register_object("script.builder.tir.Grid")
class GridForFrame(Object):
    def __init__(self, 
        extent):
        self.__init_handle_by_constructor__(
            _ffi_api.GridForFrame,
            extent,
        )