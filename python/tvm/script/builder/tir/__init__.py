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
from .block_frame import block, where, reads, writes, alloc_buffer, block_attr, init
from .for_frame import (
    ForFrame,
    grid,
    parallel,
    serial,
    thread_binding,
    unroll,
    vectorized,
)
from .prim_func_frame import (
    arg,
    func_attr,
    func_name,
    func_ret,
    match_buffer,
    preflattened_buffer,
    prim_func,
)
from .var import Buffer, buffer_decl, var, Ptr, iter_var
from .stmt import (
    Assert,
    let,
    allocate,
    allocate_const,
    launch_thread,
    realize,
    attr,
    while_,
    if_,
    then_,
    else_,
    env_thread,
    buffer_store,
    prefetch,
    evaluate,
)

from tvm.tir.expr import Broadcast as broadcast, Ramp as ramp, Shuffle
from tvm.tir.generic import cast
from tvm.tir import type_annotation
from tvm.tir.op import tvm_struct_get
from tvm.target import Target as target
from tvm.target.codegen import llvm_lookup_intrinsic_id

from .op import abs, acos, acosh, address_of, asin, asinh, atan, atan2, atanh
from .op import ceil, ceildiv, clz, copysign, cos, cosh, erf, exp, exp2, exp10
from .op import floor, floordiv, floormod, fmod, hypot
from .op import if_then_else, infinity, isfinite, isinf, isnan, isnullptr
from .op import ldexp, likely, log, log1p, log2, log10, lookup_param
from .op import max_value, min_value, nearbyint, nextafter, popcount, power
from .op import q_multiply_shift, ret, reinterpret, round, rsqrt
from .op import shift_left, shift_right, sigmoid, sin, sinh, sqrt
from .op import tan, tanh, trunc, truncdiv, truncmod
from .op import tvm_access_ptr, tvm_throw_last_error
from .op import tvm_stack_alloca, tvm_stack_make_shape, tvm_stack_make_array
from .op import call_packed, call_cpacked, call_packed_lowered, call_cpacked_lowered
from .op import call_extern, call_intrin, call_llvm_intrin, call_llvm_pure_intrin, call_pure_extern
from .op import tvm_access_ptr, tvm_tuple, tvm_struct_set, tvm_thread_allreduce
from .op import tvm_load_matrix_sync, tvm_mma_sync, tvm_bmma_sync
from .op import tvm_fill_fragment, tvm_store_matrix_sync
from .op import ptx_mma, ptx_mma_sp, ptx_ldmatrix
from .op import ptx_cp_async, ptx_wait_group, ptx_commit_group
from .op import mma_store, mma_fill
from .op import tvm_call_packed, tvm_call_cpacked, tvm_call_packed_lowered, tvm_call_cpacked_lowered
from .op import TVMBackendAllocWorkspace, TVMBackendFreeWorkspace
from .op import int8, int16, int32, int64
from .op import uint8, uint16, uint32, uint64
from .op import float8, float16, float32, float64
from .op import boolean, handle, min, max
from .op import Select, comm_reducer
