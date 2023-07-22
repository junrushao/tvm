# pylint: disable=missing-docstring,too-many-lines
from typing import List, Optional, Sequence

from tvm import relax as rx
from tvm import tir
from tvm._ffi import register_func
from tvm.runtime import NDArray

from . import op
from .core import Effect, Module, Parameter, Tensor, get_default_dtype


@register_func("effect.print")
def _print(_, array: NDArray) -> None:
    print(f"effect.print: shape = {array.shape}, dtype = {array.dtype}, data =\n{array}")


class IOEffect(Effect):
    effect: Optional[rx.Var]

    def __init__(self):
        self.effect = None

    def emit_init(self, name_hint, builder: rx.BlockBuilder) -> List[rx.DataflowVar]:
        return [builder.emit(rx.op.null_value(), f"{name_hint}.io")]

    def create(self, name_hint: str) -> List[rx.Var]:
        assert self.effect is None
        self.effect = rx.Var(f"{name_hint}.io", struct_info=rx.ObjectStructInfo())
        return [self.effect]

    def finalize(self) -> List[rx.Var]:
        result = self.effect
        self.effect = None
        return [result]

    def print_(self, tensor: Tensor) -> None:
        self.effect = rx.BlockBuilder.current().emit(
            rx.Call(
                rx.extern("effect.print"),
                args=[self.effect, tensor._expr],  # pylint: disable=protected-access
                sinfo_args=[rx.ObjectStructInfo()],
            ),
            name_hint=self.effect.name_hint,
        )


class Linear(Module):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[str] = None,
        out_dtype: Optional[str] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.weight = Parameter((out_features, in_features), dtype)
        if bias:
            self.bias = Parameter((out_features,), dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=invalid-name
        # x: [*B, in_features]
        # w: [in_features, out_features]
        w = op.permute_dims(self.weight)  # pylint: disable=invalid-name
        # x: [*B, out_features]
        x = op.matmul(x, w, out_dtype=self.out_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x


class RMSNorm(Module):
    def __init__(
        self,
        hidden_size: int,
        epsilon: float = 1e-5,
        bias: bool = True,
        dtype: Optional[str] = None,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.weight = Parameter((hidden_size,), dtype=dtype)
        if bias:
            self.bias = Parameter((hidden_size,), dtype=dtype)
        else:
            self.bias = None
        # TODO(@junrushao): add bias

    # pylint: disable=invalid-name
    def forward(self, x: Tensor):
        from tvm import te  # pylint: disable=import-outside-toplevel

        def rms_norm(x: te.Tensor, weight: te.Tensor):
            is_float32 = x.dtype == "float32"

            def f_square(x):
                return tir.Cast("float32", x) * tir.Cast("float32", x) if not is_float32 else x * x

            k = te.reduce_axis((0, x.shape[2]), name="k")
            square_sum = te.compute(
                (x.shape[0], x.shape[1]),
                lambda bsz, i: te.sum(f_square(x[bsz, i, k]), axis=k),
                name=x.op.name + "red_temp",
            )

            def f_div_cast(bsz, i, k):
                x_val = x[bsz, i, k]
                if not is_float32:
                    x_val = tir.Cast("float32", x_val)
                return x_val / tir.sqrt(square_sum[bsz, i] / x.shape[2] + self.epsilon)

            def f_mul_cast(x, y):
                value = x * y
                if not is_float32:
                    value = tir.Cast(x.dtype, value)
                return value

            return te.compute(
                x.shape,
                lambda bsz, i, k: f_mul_cast(weight(k), f_div_cast(bsz, i, k)),
                name="rms_norm",
            )

        if self.bias is None:
            return self.tensor_expr_op(rms_norm, "rms_norm", [x, self.weight])
        return op.rms_norm(x, weight=self.weight, bias=None, axes=-1, epsilon=self.epsilon)

    # pylint: enable=invalid-name


class KVCache(Effect):
    init_seq_len: int
    unit_shape: List[int]
    dtype: str
    cache: Optional[rx.Var]

    def __init__(
        self,
        init_seq_len: int,
        unit_shape: Sequence[int],
        dtype: Optional[str] = None,
    ):
        if dtype is None:
            dtype = get_default_dtype()
        # Usually the shape is: [init_seq_len, num_heads, head_dim]
        # and unit_shape = [num_heads, head_dim]
        self.init_seq_len = init_seq_len
        self.unit_shape = [int(i) for i in unit_shape]
        self.dtype = dtype

    def emit_init(self, name_hint: str, bb: rx.BlockBuilder):  # pylint: disable=arguments-renamed
        init_shape = rx.ShapeExpr([self.init_seq_len] + self.unit_shape)
        return [
            bb.emit(
                rx.Call(
                    rx.extern("vm.builtin.attention_kv_cache_create"),
                    args=[rx.op.zeros(init_shape, self.dtype), init_shape, rx.PrimValue(0)],
                    sinfo_args=[rx.ObjectStructInfo()],
                ),
                name_hint=name_hint,
            )
        ]

    def create(self, name_hint: str) -> rx.Var:
        self.cache = rx.Var(name_hint, struct_info=rx.ObjectStructInfo())
        return [self.cache]

    def finalize(self) -> List[rx.Var]:
        result = self.cache
        self.cache = None
        return [result]

    def to(self, dtype: Optional[str] = None) -> None:
        if dtype is not None:
            self.dtype = dtype

    def view(self, seq_len: tir.Var) -> Tensor:
        shape = rx.ShapeExpr([seq_len] + self.unit_shape)
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.Call(
                    rx.extern("vm.builtin.attention_kv_cache_view"),
                    args=[self.cache, shape],
                    sinfo_args=[rx.TensorStructInfo(shape, self.dtype)],
                )
            )
        )

    def append(self, new_element: Tensor) -> None:
        if new_element.dtype != self.dtype:
            raise TypeError(
                f'KVCache has been set to use dtype "{self.dtype}", '
                f'but got "{new_element.dtype}"'
            )
        self.cache = rx.BlockBuilder.current().emit(
            rx.Call(
                rx.extern("vm.builtin.attention_kv_cache_append"),
                args=[self.cache, new_element._expr],  # pylint: disable=protected-access
                sinfo_args=[rx.ObjectStructInfo()],
            )
        )


class Embedding(Module):
    def __init__(self, num: int, dim: int, dtype: Optional[str] = None):
        self.num = num
        self.dim = dim
        self.weight = Parameter((num, dim), dtype=dtype)

    def forward(self, x: Tensor):  # pylint: disable=invalid-name
        if x.ndim == 1:
            return op.take(self.weight, x, axis=0)
        return op.reshape(
            op.take(
                self.weight,
                op.reshape(x, shape=[-1]),
                axis=0,
            ),
            shape=[*x.shape, self.dim],  # TODO(@junrushao): revisit and remove self.dim
        )
