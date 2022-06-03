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
# pylint: disable=unused-import
"""Namespace for the TVMScript TIR Builder API."""

from . import axis
from .base import TIRFrame
from .block_frame import block
from .for_frame import (
    ForFrame,
    grid,
    parallel,
    serial,
    thread_binding,
    unroll,
    vectorized,
)
from .prim_func_frame import arg, prim_func
from .var import Buffer

from tvm.script.tir import (
    uint8,
    int8,
    int16,
    int32,
    int64,
    float16,
    float32,
    float64,
    boolean,
    Ptr,
)
from tvm.script.tir.intrin import comm_reducer
from tvm.tir import (
    Cast as cast,
    Select,
    Ramp as ramp,
    Broadcast as broadcast,
    Shuffle as shuffle,
    Call,
)

from tvm.tir import exp, exp2, exp10, log, log2, log10, log1p, ldexp, clz
from tvm.tir import sin, sinh, asin, asinh
from tvm.tir import cos, cosh, acos, acosh
from tvm.tir import tan, tanh, atan, atan2, atanh
from tvm.tir import erf, sigmoid, sqrt, rsqrt, floor, ceil, hypot
from tvm.tir import (
    trunc,
    abs,
    round,
    nextafter,
    nearbyint,
    power,
    popcount,
    fmod,
    if_then_else,
)
from tvm.tir import isnan, isfinite, isinf, copysign
from tvm.tir import div, indexdiv, indexmod, truncdiv, truncmod, floordiv, floormod
from tvm.tir import min, max, sum
