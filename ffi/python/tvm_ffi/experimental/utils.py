from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

if TYPE_CHECKING:
    from .type_info import DataclassField, PyTypeField, PyTypeInfo


_MISSING = type("_MISSING", (), {})()


def get_parent_type_info(type_cls: type) -> PyTypeInfo:
    for base in type_cls.__bases__:
        if (info := getattr(base, "_tvm_type_info", None)) is not None:
            return info
    from ..core import Object  # type: ignore

    return Object._tvm_type_info


def extract_dataclass_field(
    type_cls: type,
    type_field: PyTypeField,
    parent_type_info: PyTypeInfo,
) -> DataclassField:
    from .type_info import _MISSING, DataclassField, field

    field_name = type_field.name
    rhs: Any = getattr(type_cls, field_name, _MISSING)
    if isinstance(rhs, property):
        # `type_cls.{field_name}` is a property, i.e. has getter/setter.
        # It means it's probably already defined in its parent class.
        for parent_field in parent_type_info.fields:
            if parent_field.name == field_name:
                rhs = parent_field.dataclass_field
                break
    if rhs is _MISSING:
        rhs = field()
    elif isinstance(rhs, (int, float, str, bool, type(None))):
        rhs = field(default=rhs)
    elif isinstance(rhs, DataclassField):
        rhs = rhs
    else:
        raise ValueError(f"Cannot recognize field: {type_field.name}: {rhs}")
    rhs.name = type_field.name
    return rhs


def field(
    *,
    default: Any = _MISSING,
    default_factory: Any = _MISSING,
) -> DataclassField:
    if default is not _MISSING and default_factory is not _MISSING:
        raise ValueError("Cannot specify both `default` and `default_factory`")
    if default is not _MISSING:

        def _default_factory():
            return default

        default_factory = _default_factory
    return DataclassField(
        default_factory=default_factory,
        name=None,
    )


def method_init(type_cls: type, fields: list[PyTypeField]) -> Callable[..., None]:
    """Construct a fast ``__init__`` for FFI-backed dataclass-like types.

    This factory builds and returns a Python ``__init__`` function for ``type_cls``
    based on the provided ``fields``. The generated initializer mirrors a
    dataclass-style constructor while delegating the actual construction to the
    underlying FFI hook ``creator(*args)``. After successful initialization,
    it invokes ``self.__post_init__()`` if that method exists.

    Parameters
    ==========
    - type_cls: The class for which the initializer is being created. Only used to
      annotate error messages and the displayed signature string.
    - fields: Ordered list of field descriptors. Each field must have a ``name``
      and may carry a ``default_factory``; a missing factory indicates a required
      parameter.

    Returns
    =======
    - A callable suitable to be assigned to ``type_cls.__init__``.

    Notes
    =====
    - All parameters are created as ``POSITIONAL_OR_KEYWORD`` to match Python
      dataclass ergonomics while keeping call overhead low.
    - Type annotations for parameters are set to ``Any``; callers should rely on
      higher-level schema/type checks when available.
    """

    class DefaultFactory(NamedTuple):
        """Wrapper that marks a parameter as having a default factory."""

        fn: Callable[[], Any]

    annotations: dict[str, Any] = {"return": None}
    # Step 1. Split the parameters into two groups to ensure that
    # those without defaults appear first in the signature.
    params_without_defaults: list[inspect.Parameter] = []
    params_with_defaults: list[inspect.Parameter] = []
    ordering = [0] * len(fields)
    for i, field in enumerate(fields):
        assert field.name is not None
        name: str = field.name
        annotations[name] = Any  # NOTE: We might be able to handle annotations better
        default_factory = (
            field.dataclass_field.default_factory if field.dataclass_field else _MISSING
        )
        if default_factory is _MISSING:
            ordering[i] = len(params_without_defaults)
            params_without_defaults.append(
                inspect.Parameter(
                    name=name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
        else:
            ordering[i] = -len(params_with_defaults) - 1
            params_with_defaults.append(
                inspect.Parameter(
                    name=name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=DefaultFactory(fn=default_factory),
                )
            )
    for i, order in enumerate(ordering):
        if order < 0:
            ordering[i] = len(params_without_defaults) - order - 1
    # Step 2. Create the signature object
    sig = inspect.Signature(
        parameters=[
            *params_without_defaults,
            *params_with_defaults,
        ],
    )
    signature_str = (
        f"{type_cls.__module__}.{type_cls.__qualname__}.__init__("
        + ", ".join(p.name for p in sig.parameters.values())
        + ")"
    )

    # Step 3. Create the `binding` method that reorders parameters
    def touch_arg(x):
        return x.fn() if isinstance(x, DefaultFactory) else x

    def bind_args(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        args = bound.args
        args = tuple(touch_arg(args[i]) for i in ordering)
        return args

    def __init__(self: type, *args: Any, **kwargs: Any) -> None:
        e = None
        try:
            args = bind_args(*args, **kwargs)
            del kwargs
            raise NotImplementedError("FFI integration not yet implemented")
            # # Initialize underlying FFI object via reflection constructor
            # from .. import _ffi_api as _ffi_api  # local import to avoid module-level cost

            # # Fetch type key and pack arguments as (type_key, k1, v1, k2, v2, ...)
            # type_key = type_cls._tvm_type_info.type_key  # type: ignore[attr-defined]
            # packed_args: list[Any] = [type_key]
            # for f, v in zip(fields, args):
            #     # f.name is guaranteed non-None by construction above
            #     packed_args.append(f.name)  # type: ignore[arg-type]
            #     packed_args.append(v)
            # # Delegate construction to FFI and set handle on self
            # self.__init_handle_by_constructor__(  # type: ignore[attr-defined]
            #     _ffi_api.MakeObjectFromPackedArgs, *packed_args
            # )
        except Exception as _e:
            e = TypeError(f"Error in `{signature_str}`: {_e}").with_traceback(_e.__traceback__)
        if e is not None:
            raise e
        try:
            fn_post_init = self.__post_init__  # type: ignore[attr-defined]
        except AttributeError:
            pass
        else:
            fn_post_init()

    __init__.__signature__ = sig  # type: ignore[attr-defined]
    __init__.__annotations__ = annotations
    return __init__
