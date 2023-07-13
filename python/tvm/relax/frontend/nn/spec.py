# pylint: disable=missing-docstring,too-many-lines
import inspect
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

from tvm import tir
from tvm.ir import IRModule

from ... import expr as rx
from ...block_builder import BlockBuilder
from ...struct_info import ShapeStructInfo
from . import core


class Int:  # pylint: disable=too-few-public-methods
    pass


class Tensor:  # pylint: disable=too-few-public-methods
    shape: List[Union[int, str]]
    dtype: str

    def __init__(self, shape: Sequence[Union[int, str]], dtype: str) -> None:
        self.shape = list(shape)
        self.dtype = dtype


class Method:  # pylint: disable=too-few-public-methods
    name: str
    args: List[Union[Tensor, tir.Var]]
    method: Callable

    def __init__(
        self,
        name: str,
        args: List[Union[core.Tensor, tir.Var]],
        method: Callable,
    ) -> None:
        for arg in args:
            if isinstance(arg, tir.Var):
                pass
            elif isinstance(arg, core.Tensor):
                for shape in arg.shape:
                    assert isinstance(shape, (tir.Var, int))
            else:
                raise TypeError(f"Invalid argument type: {type(arg)}")
        self.name = name
        self.args = args
        self.method = method

    @staticmethod
    def from_raw(module: core.Module, name: str, spec: Dict[str, Union[Int, Tensor]]) -> "Method":
        if not hasattr(module, name):
            raise AttributeError(f"Method {name} not found in module {module}")
        str2var: Dict[str, tir.Var] = {}

        def _get_var(name: str) -> tir.Var:
            if name in str2var:
                return str2var[name]
            var = tir.Var(name, "int64")
            str2var[name] = var
            return var

        method = getattr(module, name, None)
        method_signature = inspect.signature(method)
        arg_names = list(method_signature.parameters.keys())
        args = []
        for arg_name in arg_names:
            if arg_name not in spec:
                raise ValueError(f"Argument {arg_name} not found in the spec of method {name}")
            arg_spec = spec[arg_name]
            if arg_spec is Int:
                arg_spec = arg_spec()
            if isinstance(arg_spec, Int):
                args.append(_get_var(arg_name))
            elif isinstance(arg_spec, Tensor):
                args.append(
                    core._tensor_placeholder(  # pylint: disable=protected-access
                        name=arg_name,
                        shape=[_get_var(x) if isinstance(x, str) else x for x in arg_spec.shape],
                        dtype=arg_spec.dtype,
                    )
                )
            else:
                raise TypeError(f"Invalid argument spec in method {name}: {arg_spec}")
        return Method(name, args, method)

    @staticmethod
    def from_torch(
        module: core.Module,
        name: str,
        args: List[Any],
    ) -> "Method":
        from .torch import _spec_from_torch  # pylint: disable=import-outside-toplevel

        return _spec_from_torch(module, name, args)

    def __repr__(self) -> str:
        args: List[str] = []
        for arg in self.args:
            if isinstance(arg, tir.Var):
                args.append(f"{arg.name} : int")
            elif isinstance(arg, core.Tensor):
                args.append(f'{arg._expr.name_hint} : Tensor({arg.shape}, "{arg.dtype}")')
            else:
                raise TypeError(f"Invalid argument type: {arg}")
        return f"{self.name}({', '.join(args)})"


class Module:  # pylint: disable=too-few-public-methods
    module: core.Module
    methods: Dict[str, Method]

    def __init__(self, module: core.Module, methods: Dict[str, Method]):
        self.module = module
        self.methods = methods

    @staticmethod
    def from_raw(
        module: core.Module,
        spec: Dict[str, Dict[str, Union["Spec.Int", "Spec.Tensor"]]],
    ) -> "Module":
        methods = OrderedDict()
        for method_name, method_spec in spec.items():
            methods[method_name] = Method.from_raw(module, method_name, method_spec)
        return Module(module, methods)

    def __repr__(self) -> str:
        return "Module:\n" + "\n".join([f"  {method}" for method in self.methods])


class SpecBuilder:
    _tls = threading.local()

    builder: BlockBuilder
    io_effect: core.Effect

    def __init__(self) -> None:
        from .modules import IOEffect  # pylint: disable=import-outside-toplevel

        self.builder = BlockBuilder()
        self.io_effect = IOEffect()

    @staticmethod
    def current() -> "SpecBuilder":
        assert hasattr(SpecBuilder._tls, "current")
        return SpecBuilder._tls.current

    def __enter__(self) -> "SpecBuilder":
        assert not hasattr(SpecBuilder._tls, "current")
        SpecBuilder._tls.current = self
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert hasattr(SpecBuilder._tls, "current")
        delattr(SpecBuilder._tls, "current")

    def build(self, spec: Module) -> IRModule:
        _check_parameter_binding(spec.module)

        effects = [
            ("", self.io_effect),
        ] + list(
            core._attribute_finder(  # pylint: disable=protected-access
                spec.module,
                "",
                condition_yield=lambda x: isinstance(x, core.Effect),
            )
        )
        with self:
            with self.builder.function("_initialize_effect"):
                with self.builder.dataflow():
                    outputs = _emit_effect_init(self.builder, effects)
                self.builder.emit_func_output(outputs, params=[])
            for method_spec in spec.methods.values():
                with self.builder.function(method_spec.name):
                    with self.builder.dataflow():
                        params = _spec_to_params(method_spec)
                        outputs, effect_inputs = _emit_method(self.builder, method_spec, effects)
                    self.builder.emit_func_output(outputs, params=params + effect_inputs)
        return self.builder.get()


def _emit_effect_init(
    builder: BlockBuilder,
    effects: List[Tuple[str, core.Effect]],
):
    outputs = []
    for prefix, effect in effects:
        inits = effect.emit_init(prefix, builder)
        assert isinstance(inits, list)
        outputs.extend(inits)
    outputs = builder.emit_output(builder.emit(rx.Tuple(outputs)))
    return outputs


def _emit_method(
    builder: BlockBuilder,
    spec: Module,
    effects: List[Tuple[str, core.Effect]],
):
    effect_inputs = []
    for prefix, effect in effects:
        effect_inputs.extend(effect.create(prefix))
    outputs = spec.method(*spec.args)
    effect_outputs = []
    for _, effect in effects:
        effect_outputs.extend(effect.finalize())
    outputs = builder.emit_output(
        rx.Tuple(
            [
                core._unwrap_nested(outputs),  # pylint: disable=protected-access
                rx.Tuple(effect_outputs),
            ]
        )
    )
    return outputs, effect_inputs


def _spec_to_params(spec: Method):
    params = []
    for arg in spec.args:
        if isinstance(arg, core.Tensor):
            params.append(arg._expr)  # pylint: disable=protected-access
        elif isinstance(arg, tir.Var):
            params.append(rx.Var(arg.name, struct_info=ShapeStructInfo(values=[arg])))
        else:
            raise ValueError(f"Unsupported argument type {type(arg)}")
    return params


def _check_parameter_binding(model: core.Module):
    missing_parameters = []
    for name, param in model.named_parameters():
        if not isinstance(param._expr, rx.Constant):  # pylint: disable=protected-access
            missing_parameters.append(name)
    if missing_parameters:
        raise ValueError(f"Parameters {missing_parameters} are not set to any concrete values. ")
