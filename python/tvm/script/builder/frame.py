from tvm._ffi import register_object as _register_object

from tvm.runtime import Object

from . import _ffi_api



@_register_object("script.builder.Frame")
class Frame(Object):
    def __init__(self) -> None:
        pass

    def __enter__(self) -> None:
        _ffi_api.EnterFrame(self)

    def __exit__(self, ptype, value, trace) -> None:
        _ffi_api.ExitFrame(self)


@_register_object("script.builder.IRModuleFrame")
class IRModuleFrame(Frame):
    def __init__(self) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.IRModuleFrame
        )

    def __enter__(self) -> None:
        _ffi_api.EnterFrame(self)

    def __exit__(self, ptype, value, trace) -> None:
        _ffi_api.ExitFrame(self)