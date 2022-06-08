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
"""TVM Script TIR Op"""

from . import _ffi_api


boolean = _ffi_api.PrimType("bool")

int8 = _ffi_api.PrimType("int8")
int16 = _ffi_api.PrimType("int16")
int32 = _ffi_api.PrimType("int32")
int64 = _ffi_api.PrimType("int64")

uint8 = _ffi_api.PrimType("uint8")
uint16 = _ffi_api.PrimType("uint16")
uint32 = _ffi_api.PrimType("uint32")
uint64 = _ffi_api.PrimType("uint64")

float8 = _ffi_api.PrimType("float8")
float16 = _ffi_api.PrimType("float16")
float32 = _ffi_api.PrimType("float32")
float64 = _ffi_api.PrimType("float64")

from tvm.tir.op import abs, popcount, nextafter, copysign, fmod
from tvm.tir.op import (
    floor,
    floordiv,
    floormod,
    ceil,
    round,
    trunc,
    truncdiv,
    truncmod,
    nearbyint,
)
from tvm.tir.op import (
    hypot,
    ldexp,
    power,
    exp,
    exp2,
    exp10,
    erf,
    sqrt,
    rsqrt,
    log,
    log2,
    log10,
    log1p,
    sigmoid,
)
from tvm.tir.op import isnan, isfinite, isinf
from tvm.tir.op import cos, cosh, sin, sinh, tan, tanh
from tvm.tir.op import acos, acosh, asin, asinh, atan, atanh
from tvm.tir.op import atan2, clz, comm_reducer, infinity, reinterpret
from tvm.tir.op import min_value, max_value, if_then_else
from tvm.tir.op import call_packed, call_extern
from tvm.tir.expr import Select, Ramp, Broadcast, Shuffle
from tvm.tir.generic import cast


def min(a, b, span=None):
    """Compute the minimum value of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api.min(a, b, span)  # type: ignore


def max(a, b, span=None):
    """Compute the maximum value of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api.max(a, b, span)  # type: ignore
