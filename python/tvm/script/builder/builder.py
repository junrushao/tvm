from tvm._ffi import register_object as _register_object

from tvm.runtime import Object

from . import _ffi_api


@_register_object("script.builder.Builder")
class Builder(Object):
    def __init__(self) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Builder
        )

    def __enter__(self) -> None:
        _ffi_api.EnterBuilder(self)

    def __exit__(self, ptype, value, trace) -> None:
        _ffi_api.ExitBuilder(self)

    @staticmethod
    def current(self):
        return _ffi_api.CurrentBuilder(self)

    def get(self):
        return _ffi_api.BuilderGet(self)


def def_(name, var):
    print(dir(_ffi_api))
    return _ffi_api.Def(name, var)

def def_many(names, vars):
    assert len(names) == len(vars)
    return [def_(name, var) for name, var in zip(names, vars)]