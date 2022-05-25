from tvm._ffi import register_object as _register_object

from tvm.runtime import Object

from . import _ffi_api


@_register_object("script.builder.Builder")
class Builder(Object):
    def __init__(self) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.builder
        )

    def __enter__(self) -> None:
        _ffi_api.enter(self)

    def __exit__(self, ptype, value, trace) -> None:
        _ffi_api.exit(self)

    def currentBuilder(self) -> 'Builder':
        return _ffi_api.current(self)
