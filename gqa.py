# pylint: disable=missing-docstring,invalid-name
import numpy as np
import torch

from tvm import relax as rx
from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op

# Module, Tensor, repeat, spec


class GroupedQueryAttention(nn.Module):
    def __init__(self, n_rep: int):
        self.n_rep = n_rep

    def forward(self, x: nn.Tensor):
        if self.n_rep == 1:
            return x
        x = op.repeat(x, self.n_rep, axis=2)
        nn.print_(x)
        return x


def torch_method(x: torch.Tensor, n_rep: int):
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def main():
    batch_size = 1
    seq_len = 64
    n_kv_heads = 8
    head_dim = 128
    n_rep = 8

    model = GroupedQueryAttention(n_rep=n_rep)
    torch_model = model.jit(
        spec={
            "forward": {
                "x": nn.spec.Tensor((batch_size, seq_len, n_kv_heads, head_dim), dtype="float32"),
            }
        },
        target="llvm",
        device="cpu",
        out_format="torch",
    )
    x = torch.from_numpy(
        np.random.rand(batch_size, seq_len, n_kv_heads, head_dim).astype("float32")
    )
    relax_y = torch_model["forward"](x)
    torch_y = torch_method(x, n_rep)
    print((relax_y == torch_y).all())


if __name__ == "__main__":
    main()
