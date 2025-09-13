import typing
from collections.abc import Callable
from typing import get_type_hints

from . import utils
from .type_info import DataclassField, PyTypeField, PyTypeInfo

try:
    from typing import dataclass_transform
except ImportError:
    from typing_extensions import dataclass_transform

InputClsType = typing.TypeVar("InputClsType")


@dataclass_transform(field_specifiers=(utils.field, DataclassField))
def c_class(
    type_key: str,
    init: bool = True,
) -> Callable[[type[InputClsType]], type[InputClsType]]:
    def decorator(super_type_cls: type[InputClsType]) -> type[InputClsType]:
        # Step 1. Retrieve `type_info` from registry
        parent_type_info: PyTypeInfo = utils.get_parent_type_info(super_type_cls)
        type_info: PyTypeInfo = PyTypeInfo.register_c_class_on_py(type_key)
        # Step 2. Reflect all the fields of the type
        type_info.fields = inspect_c_class_fields(super_type_cls, type_info, parent_type_info)
        # Step 3. Create the proxy class with the fields as properties
        type_cls: type[InputClsType] = type_info.create(
            cls=super_type_cls,
            methods={
                "__init__": utils.method_init(super_type_cls, fields=type_info.fields)
                if init
                else None
            },
        )
        return type_cls

    return decorator


def inspect_c_class_fields(
    type_cls: type,
    type_info: PyTypeInfo,
    parent_type_info: PyTypeInfo,
) -> list[PyTypeField]:
    type_hints = get_type_hints(type_cls)
    type_fields_reflected: dict[str, PyTypeField] = {f.name: f for f in type_info.fields}
    type_fields: list[PyTypeField] = []
    for type_field in parent_type_info.fields:
        field_name: str = type_field.name
        type_field: PyTypeField = type_fields_reflected.pop(field_name, None)
        if type_field is None or type_hints.pop(field_name, None) is None:
            raise ValueError(
                f"Missing field `{type_cls}.{field_name}`. Defined in C but not in Python"
            )
        type_fields.append(type_field)
    for field_name, _field_ty_py in type_hints.items():
        if not field_name.startswith("_tvm_"):
            type_field: PyTypeField = type_fields_reflected.pop(field_name, None)
            if type_field is None:
                raise ValueError(
                    f"Extraneous field `{type_cls}.{field_name}`. Defined in Python but not in C"
                )
            type_fields.append(type_field)
    if type_fields_reflected:
        extra_fields = ", ".join(f"`{f.name}`" for f in type_fields_reflected.values())
        raise ValueError(
            f"Missing fields in `{type_cls}`: {extra_fields}. Defined in C but not in Python"
        )
    for type_field in type_fields:
        type_field.dataclass_field = utils.extract_dataclass_field(
            type_cls, type_field, parent_type_info
        )
    return type_fields
