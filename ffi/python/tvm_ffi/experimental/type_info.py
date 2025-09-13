from __future__ import annotations

import dataclasses
import functools
import sys
from typing import Any, Callable, ClassVar, TypeVar

_DATACLASS_SLOTS_ON = {"slots": True} if sys.version_info >= (3, 10) else {}
_InputClsType = TypeVar("_InputClsType")


@dataclasses.dataclass
class DataclassField:
    _: dataclasses.KW_ONLY
    default_factory: Callable[[], Any]
    name: str | None


@dataclasses.dataclass(eq=False, **_DATACLASS_SLOTS_ON)
class PyTypeField:
    name: str
    doc: str
    size: int
    offset: int
    frozen: bool
    getter: Any
    setter: Any
    dataclass_field: DataclassField | None = None

    def create(self, cls: type) -> property:
        name = self.name

        def fget(this: Any, _name: str = name) -> Any:
            return self.getter(this)  # type: ignore[misc]

        def fset(this: Any, value: Any, _name: str = name) -> None:
            self.setter(this, value)  # type: ignore[misc]

        fget.__name__ = fset.__name__ = name
        fget.__module__ = fset.__module__ = cls.__module__
        fget.__qualname__ = fset.__qualname__ = f"{cls.__qualname__}.{name}"  # type: ignore[attr-defined]
        fget.__doc__ = fset.__doc__ = f"Property `{self.name}` of class `{cls.__qualname__}`"  # type: ignore[attr-defined]
        return property(
            fget=fget if self.getter else None,
            fset=fset if (not self.frozen) and self.setter else None,
            doc=f"{cls.__module__}.{cls.__qualname__}.{name}",
        )


@dataclasses.dataclass(eq=False, **_DATACLASS_SLOTS_ON)
class PyTypeMethod:
    name: str
    doc: str
    func: Any
    is_static: bool


@dataclasses.dataclass(eq=False, **_DATACLASS_SLOTS_ON)
class PyTypeInfo:
    _registry: ClassVar[dict[str, "PyTypeInfo"]] = {}

    type_cls: type | None
    type_index: int
    type_key: str
    fields: list[PyTypeField]
    methods: list[PyTypeMethod]

    @staticmethod
    def register_c_class_on_py(type_key: str) -> "PyTypeInfo":
        from ..core import _object_type_key_to_index as key2idx  # type: ignore
        from ..core import _type_index_to_py_type_info as idx2info  # type: ignore

        if type_key in PyTypeInfo._registry:
            raise ValueError(f"Type is already registered: {type_key}")

        info: PyTypeInfo = idx2info(key2idx(type_key))
        PyTypeInfo._registry[type_key] = info
        return info

    def create(
        self,
        cls: type[_InputClsType],
        methods: dict[str, Callable[..., Any] | None],
    ) -> type[_InputClsType]:
        assert self.type_cls is None, "Type class is already created"
        cls_name = cls.__name__
        cls_bases = cls.__bases__
        if cls_bases == (object,):
            # If the class inherits from `object`, we need to set the base class to `Object`
            from ..core import Object  # type: ignore

            cls_bases = (Object,)

        attrs = dict(cls.__dict__)
        attrs.pop("__dict__", None)
        attrs.pop("__weakref__", None)
        attrs["__slots__"] = ()
        attrs["_tvm_type_info"] = self
        for field in self.fields:
            attrs[field.name] = field.create(cls)
        for name, method in methods.items():
            if method is not None:
                method.__module__ = cls.__module__
                method.__name__ = name
                method.__qualname__ = f"{cls.__qualname__}.{name}"
                method.__doc__ = f"Method `{name}` of class `{cls.__qualname__}`"
                attrs[name] = method

        new_cls = type(cls_name, cls_bases, attrs)
        new_cls.__module__ = cls.__module__
        new_cls = functools.wraps(cls, updated=())(new_cls)  # type: ignore
        self.type_cls = new_cls
        return new_cls
