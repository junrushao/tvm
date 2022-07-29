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
# pylint: disable=unused-import, redefined-builtin
"""Namespace for Tensor-level IR"""
from tvm.ir import PrimExpr
from tvm.runtime import const

from .buffer import Buffer, DataProducer, decl_buffer
from .data_layout import BijectiveLayout, Layout, bijective_layout, layout
from .expr import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    Add,
    And,
    Any,
    Broadcast,
    BufferLoad,
    Call,
    CallEffectKind,
    Cast,
    CommReducer,
    Div,
    FloatImm,
    FloorDiv,
    FloorMod,
    IntImm,
    IterVar,
    Let,
    Load,
    Max,
    Min,
    Mod,
    Mul,
    Not,
    Or,
    ProducerLoad,
    Ramp,
    Reduce,
    Select,
    Shuffle,
    SizeVar,
    StringImm,
    Sub,
    Var,
)

from .stmt import (
    Allocate,
    AllocateConst,
    AssertStmt,
    AttrStmt,
    Block,
    BlockRealize,
    BufferRealize,
    BufferRegion,
    BufferStore,
    Evaluate,
    For,
    ForKind,
    IfThenElse,
    LetStmt,
    MatchBufferRegion,
    Prefetch,
    ProducerRealize,
    ProducerStore,
    SeqStmt,
    Stmt,
    Store,
    While,
    stmt_list,
    stmt_seq,
    type_annotation,
)

from .stmt import ProducerRealize, SeqStmt
from .stmt import IfThenElse, Evaluate, Prefetch, stmt_seq, stmt_list
from .stmt import BufferRegion, MatchBufferRegion, Block, BlockRealize, type_annotation

from .function import PrimFunc, TensorIntrin, IndexMap

from .op import call_packed, call_cpacked, call_packed_lowered, call_cpacked_lowered
from .op import call_intrin, call_pure_extern, call_extern
from .op import call_llvm_intrin, call_llvm_pure_intrin, ret, all, any, min_value, max_value, trace
from .op import tvm_access_ptr, tvm_throw_last_error
from .op import tvm_stack_alloca, tvm_stack_make_shape, tvm_stack_make_array
from .op import exp, exp2, exp10, log, log2, log10, log1p, ldexp, clz
from .op import sin, sinh, asin, asinh
from .op import cos, cosh, acos, acosh
from .op import tan, tanh, atan, atan2, atanh
from .op import erf, sigmoid, sqrt, rsqrt, floor, ceil, hypot
from .op import trunc, abs, round, nextafter, nearbyint, power, popcount, fmod, if_then_else
from .op import isnan, isfinite, isinf, copysign, isnullptr
from .op import div, indexdiv, indexmod, truncdiv, truncmod, floordiv, floormod, ceildiv
from .op import comm_reducer, min, max, sum
from .op import q_multiply_shift, address_of, lookup_param, shift_left, shift_right
from .op import likely
from .op import ptx_mma, ptx_mma_sp, ptx_ldmatrix, ptx_cp_async, ptx_wait_group, ptx_commit_group
from .op import mma_store, mma_fill
from .op import tvm_access_ptr, tvm_tuple, tvm_struct_get, tvm_struct_set
from .op import tvm_thread_allreduce, tvm_load_matrix_sync, tvm_mma_sync, tvm_bmma_sync
from .op import tvm_fill_fragment, tvm_store_matrix_sync
from .op import TVMBackendAllocWorkspace, TVMBackendFreeWorkspace

from .schedule import StmtSRef, BlockScope, ScheduleState, Schedule, ScheduleError

from . import schedule
from . import ir_builder
from . import transform
from . import analysis
from . import stmt_functor
from . import usmp
