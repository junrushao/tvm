# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-module-docstring,missing-function-docstring
from typing import Tuple
from tvm import te, tir, topi


def batch_matmul_nkkm(  # pylint: disable=invalid-name
    B: int,
    N: int,
    M: int,
    K: int,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    x = te.placeholder((B, N, K), name="X")
    y = te.placeholder((B, K, M), name="Y")
    k = te.reduce_axis((0, K), name="k")
    z = te.compute(  # pylint: disable=invalid-name
        (B, N, M),
        lambda b, i, j: te.sum(x[b][i][k] * y[b][k][j], axis=[k]),
        name="Z",
    )
    return (x, y, z)


def conv1d_nlc(  # pylint: disable=invalid-name
    N: int,
    L: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, L, CI), name="inputs")
    weight = te.placeholder((kernel_size, CI // groups, CO), name="weight")

    batch_size, in_len, _ = inputs.shape
    k_len, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups
    out_len = (in_len + 2 * padding - dilation * (k_len - 1) - 1) // stride + 1
    rc = te.reduce_axis((0, channel_per_group), name="rc")
    rl = te.reduce_axis((0, k_len), name="rl")

    padded = topi.nn.pad(inputs, [0, padding, 0])
    output = te.compute(
        (batch_size, out_len, out_channel),
        lambda n, l, co: te.sum(
            (
                padded[
                    n,
                    l * stride + rl * dilation,
                    co // out_channel_per_group * channel_per_group + rc,
                ]
                * weight[rl, rc, co]
            ),
            axis=[rl, rc],
        ),
        name="conv1d_nlc",
    )
    return (inputs, weight, output)


def conv2d_nhwc(  # pylint: disable=invalid-name
    N: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, H, W, CI), name="inputs")
    weight = te.placeholder((kernel_size, kernel_size, CI // groups, CO), name="weight")
    batch_size, in_h, in_w, _ = inputs.shape
    k_h, k_w, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups

    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    rc = te.reduce_axis((0, channel_per_group), name="rc")

    padded = topi.nn.pad(inputs, [0, padding, padding, 0])
    output = te.compute(
        (batch_size, out_h, out_w, out_channel),
        lambda n, h, w, co: te.sum(
            (
                padded[
                    n,
                    h * stride + rh * dilation,
                    w * stride + rw * dilation,
                    co // out_channel_per_group * channel_per_group + rc,
                ]
                * weight[rh, rw, rc, co]
            ),
            axis=[rh, rw, rc],
        ),
        name="conv2d_nhwc",
    )
    return (inputs, weight, output)


def conv3d_ndhwc(  # pylint: disable=invalid-name
    N: int,
    D: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, D, H, W, CI))
    weight = te.placeholder((kernel_size, kernel_size, kernel_size, CI // groups, CO))
    batch_size, in_d, in_h, in_w, _ = inputs.shape
    k_d, k_h, k_w, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups

    out_d = (in_d + 2 * padding - dilation * (k_d - 1) - 1) // stride + 1
    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rd = te.reduce_axis((0, k_d), name="rd")
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    rc = te.reduce_axis((0, channel_per_group), name="rc")

    padded = topi.nn.pad(inputs, [0, padding, padding, padding, 0])
    output = te.compute(
        (batch_size, out_d, out_h, out_w, out_channel),
        lambda n, d, h, w, co: te.sum(
            (
                padded[
                    n,
                    d * stride + rd * dilation,
                    h * stride + rh * dilation,
                    w * stride + rw * dilation,
                    co // out_channel_per_group * channel_per_group + rc,
                ]
                * weight[rd, rh, rw, rc, co]
            ),
            axis=[rd, rh, rw, rc],
        ),
        name="conv3d_ndhwc",
    )
    return (inputs, weight, output)


def depthwise_conv2d_nhwc(  # pylint: disable=invalid-name
    N: int,
    H: int,
    W: int,
    C: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    factor: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, H, W, C))
    weight = te.placeholder((factor, kernel_size, kernel_size, C))
    batch_size, in_h, in_w, in_channel = inputs.shape
    factor, k_h, k_w, in_channel = weight.shape
    out_channel = in_channel * factor
    assert factor.value == 1, "Not optimized for factor != 1"
    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    padded = topi.nn.pad(inputs, [0, padding, padding, 0])
    output = te.compute(
        (batch_size, out_h, out_w, out_channel),
        lambda n, h, w, c: te.sum(
            (
                padded[
                    n,
                    h * stride + rh * dilation,
                    w * stride + rw * dilation,
                    c // factor,
                ]
                * weight[c % factor, rh, rw, c // factor]
            ),
            axis=[rh, rw],
        ),
        name="depth_conv2d_nhwc",
    )
    return (inputs, weight, output)


def conv2d_transpose_nhwc(  # pylint: disable=invalid-name
    N: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, H, W, CI), name="inputs")
    weight = te.placeholder((kernel_size, kernel_size, CI, CO), name="weight")

    batch, in_h, in_w, in_c = inputs.shape
    filter_h, filter_w, in_c, out_c = weight.shape
    stride_h, stride_w = (stride, stride)

    # compute padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = topi.nn.get_pad_tuple(
        padding, (filter_h, filter_w)
    )
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right

    # padding stage
    padded = topi.nn.pad(
        inputs,
        [
            0,
            (bpad_top + stride_h - 1) // stride_h,
            (bpad_left + stride_w - 1) // stride_w,
            0,
        ],
        [
            0,
            (bpad_bottom + stride_h - 1) // stride_h,
            (bpad_right + stride_w - 1) // stride_w,
            0,
        ],
    )

    # remove extra padding introduced by dilatation
    idx_div = te.indexdiv
    idx_mod = te.indexmod
    border_h = idx_mod(stride_h - idx_mod(bpad_top, stride_h), stride_h)
    border_w = idx_mod(stride_w - idx_mod(bpad_left, stride_w), stride_w)

    # dilation stage
    strides = [1, stride_h, stride_w, 1]
    n = len(padded.shape)

    # We should embed this dilation directly into te.compute rather than creating a new te.compute.
    # Only in this way can we use unroll to eliminate the multiplication of zeros.
    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if not strides[i] == 1:
                index_tuple.append(idx_div(indices[i], strides[i]))
                not_zero.append(idx_mod(indices[i], strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = te.all(*not_zero)
            return te.if_then_else(not_zero, padded(*index_tuple), tir.const(0.0, padded.dtype))
        return padded(*index_tuple)

    # convolution stage
    out_h = (in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    rc = te.reduce_axis((0, in_c), name="rc")
    rh = te.reduce_axis((0, filter_h), name="rh")
    rw = te.reduce_axis((0, filter_w), name="rw")

    output = te.compute(
        (batch, out_h, out_w, out_c),
        lambda n, h, w, co: te.sum(
            _dilate(n, h + rh + border_h, w + rw + border_w, rc)
            * weight[filter_h - 1 - rh, filter_w - 1 - rw, rc, co],
            axis=[rh, rw, rc],
        ),
        name="conv2d_transpose_nhwc",
    )
    # TODO(lmzheng): add constraints on the tile size of h and w
    return (inputs, weight, output)


def conv2d_capsule_nhwijc(  # pylint: disable=invalid-name
    N: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    capsule_size: int = 4,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, H, W, capsule_size, capsule_size, CI), name="inputs")
    weight = te.placeholder(
        (kernel_size, kernel_size, capsule_size, capsule_size, CI, CO), name="weight"
    )
    batch_size, in_h, in_w, _, _, in_channel = inputs.shape
    k_h, k_w, _, _, _, out_channel = weight.shape

    out_h = (in_h + 2 * padding - kernel_size) // stride + 1
    out_w = (in_w + 2 * padding - kernel_size) // stride + 1

    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    cap_k = te.reduce_axis((0, capsule_size), name="cap_k")
    rc = te.reduce_axis((0, in_channel), name="rc")

    padded = topi.nn.pad(inputs, [0, padding, padding, 0, 0, 0])
    output = te.compute(
        (batch_size, out_h, out_w, capsule_size, capsule_size, out_channel),
        lambda n, h, w, cap_i, cap_j, co: te.sum(
            (
                padded[n, h * stride + rh, w * stride + rw, cap_i, cap_k, rc]
                * weight[rh, rw, cap_k, cap_j, rc, co]
            ),
            axis=[rh, rw, cap_k, rc],
        ),
        name="conv2d_capsule_nhwijc",
    )
    return (inputs, weight, output)


def norm_bmn(  # pylint: disable=invalid-name
    B: int,
    M: int,
    N: int,
) -> Tuple[te.Tensor, te.Tensor]:
    a = te.placeholder((B, M, N), name="A")
    i = te.reduce_axis((0, M), name="i")
    j = te.reduce_axis((0, N), name="j")
    c = te.compute(
        (B,),
        lambda b: te.sum(a[b][i][j] * a[b][i][j], axis=[i, j]),
        name="C",
    )
    d = te.compute((B,), lambda b: te.sqrt(c[b]), name="D")
    return (a, d)
