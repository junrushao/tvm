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

import inspect
from tvm.tir import op
from tvm.tir import IntImm, PrimExpr, Select as select, CommReducer, Var
from tvm.runtime import convert_to_object


def op_wrapper(func):
    def wrapped(*args, **kwargs):
        if "dtype" in kwargs:
            kwargs.pop("dtype")
        return func(*args, **kwargs)

    return wrapped


def dtype_forward(func):
    def forwarded(*args, **kwargs):
        if "dtype" in kwargs:
            args = (kwargs.pop("dtype"),) + args
        return func(*args, **kwargs)

    return forwarded


abs = op_wrapper(op.abs)
acos = op_wrapper(op.acos)
acosh = op_wrapper(op.acosh)
address_of = op_wrapper(op.address_of)
asin = op_wrapper(op.asin)
asinh = op_wrapper(op.asinh)
atan = op_wrapper(op.atan)
atan2 = op_wrapper(op.atan2)
atanh = op_wrapper(op.atanh)
ceil = op_wrapper(op.ceil)
clz = op_wrapper(op.clz)
copysign = op_wrapper(op.copysign)
cos = op_wrapper(op.cos)
cosh = op_wrapper(op.cosh)
erf = op_wrapper(op.erf)
exp = op_wrapper(op.exp)
exp2 = op_wrapper(op.exp2)
exp10 = op_wrapper(op.exp10)
floor = op_wrapper(op.floor)
floordiv = op_wrapper(op.floordiv)
floormod = op_wrapper(op.floormod)
fmod = op_wrapper(op.fmod)
hypot = op_wrapper(op.hypot)
if_then_else = op_wrapper(op.if_then_else)
infinity = op_wrapper(op.infinity)
isfinite = op_wrapper(op.isfinite)
isinf = op_wrapper(op.isinf)
isnan = op_wrapper(op.isnan)
isnullptr = op_wrapper(op.isnullptr)
ldexp = op_wrapper(op.ldexp)
likely = op_wrapper(op.likely)
log = op_wrapper(op.log)
log1p = op_wrapper(op.log1p)
log2 = op_wrapper(op.log2)
log10 = op_wrapper(op.log10)
lookup_param = op_wrapper(op.lookup_param)
max_value = op_wrapper(op.max_value)
min_value = op_wrapper(op.min_value)
nearbyint = op_wrapper(op.nearbyint)
nextafter = op_wrapper(op.nextafter)
popcount = op_wrapper(op.popcount)
power = op_wrapper(op.power)
q_multiply_shift = op_wrapper(op.q_multiply_shift)
ret = op_wrapper(op.ret)
reinterpret = dtype_forward(op.reinterpret)
round = op_wrapper(op.round)
rsqrt = op_wrapper(op.rsqrt)
shift_left = op_wrapper(op.shift_left)
shift_right = op_wrapper(op.shift_right)
sigmoid = op_wrapper(op.sigmoid)
sin = op_wrapper(op.sin)
sinh = op_wrapper(op.sinh)
sqrt = op_wrapper(op.sqrt)
tan = op_wrapper(op.tan)
tanh = op_wrapper(op.tanh)
trunc = op_wrapper(op.trunc)
truncdiv = op_wrapper(op.truncdiv)
truncmod = op_wrapper(op.truncmod)

tvm_access_ptr = op_wrapper(op.tvm_access_ptr)
tvm_throw_last_error = op_wrapper(op.tvm_throw_last_error)
tvm_stack_alloca = op_wrapper(op.tvm_stack_alloca)
tvm_stack_make_shape = op_wrapper(op.tvm_stack_make_shape)
tvm_stack_make_array = op_wrapper(op.tvm_stack_make_array)

call_packed = op_wrapper(op.call_packed)
call_cpacked = op_wrapper(op.call_cpacked)
call_packed_lowered = op_wrapper(op.call_packed_lowered)
call_cpacked_lowered = op_wrapper(op.call_cpacked_lowered)

call_extern = dtype_forward(op.call_extern)
call_intrin = dtype_forward(op.call_intrin)
call_llvm_intrin = dtype_forward(op.call_llvm_intrin)
call_llvm_pure_intrin = dtype_forward(op.call_llvm_pure_intrin)
call_pure_extern = dtype_forward(op.call_pure_extern)

tvm_access_ptr = op_wrapper(op.tvm_access_ptr)
tvm_tuple = op_wrapper(op.tvm_tuple)
tvm_struct_set = op_wrapper(op.tvm_struct_set)

tvm_thread_allreduce = op_wrapper(op.tvm_thread_allreduce)
tvm_load_matrix_sync = op_wrapper(op.tvm_load_matrix_sync)
tvm_mma_sync = op_wrapper(op.tvm_mma_sync)
tvm_bmma_sync = op_wrapper(op.tvm_bmma_sync)
tvm_fill_fragment = op_wrapper(op.tvm_fill_fragment)
tvm_store_matrix_sync = op_wrapper(op.tvm_store_matrix_sync)

ptx_mma = op_wrapper(op.ptx_mma)
ptx_mma_sp = op_wrapper(op.ptx_mma_sp)
ptx_ldmatrix = op_wrapper(op.ptx_ldmatrix)
ptx_cp_async = op_wrapper(op.ptx_cp_async)
ptx_wait_group = op_wrapper(op.ptx_wait_group)
ptx_commit_group = op_wrapper(op.ptx_commit_group)
mma_store = op_wrapper(op.mma_store)
mma_fill = op_wrapper(op.mma_fill)

tvm_call_packed = call_cpacked
tvm_call_cpacked = call_cpacked
tvm_call_packed_lowered = call_cpacked_lowered
tvm_call_cpacked_lowered = call_cpacked_lowered

from . import _ffi_api


def int8(expr=None):
    return _ffi_api.Int8(expr)


def int16(expr=None):
    return _ffi_api.Int16(expr)


def int32(expr=None):
    return _ffi_api.Int32(expr)


def int64(expr=None):
    return _ffi_api.Int64(expr)


def uint8(expr=None):
    return _ffi_api.UInt8(expr)


def uint16(expr=None):
    return _ffi_api.UInt16(expr)


def uint32(expr=None):
    return _ffi_api.UInt32(expr)


def uint64(expr=None):
    return _ffi_api.UInt64(expr)


def float8(expr=None):
    if not isinstance(expr, PrimExpr):
        expr = convert_to_object(expr)
    return _ffi_api.Float8(expr)


def float16(expr=None):
    if not isinstance(expr, PrimExpr):
        expr = convert_to_object(expr)
    return _ffi_api.Float16(expr)


def float32(expr=None):
    if not isinstance(expr, PrimExpr):
        expr = convert_to_object(expr)
    return _ffi_api.Float32(expr)


def float64(expr=None):
    if not isinstance(expr, PrimExpr):
        expr = convert_to_object(expr)
    return _ffi_api.Float64(expr)


def boolean(expr=None):
    return _ffi_api.Boolean(expr)


def handle():
    return _ffi_api.Handle()


def min(a, b):
    """Compute the minimum value of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api.min(a, b)  # type: ignore


def max(a, b):
    """Compute the maximum value of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api.max(a, b)  # type: ignore


def Select(condition, true_value, false_value):
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return select(condition, true_value, false_value)


def comm_reducer(combiner, identity):
    """Create a CommReducer from lambda inputs/outputs and the identities"""
    params = inspect.signature(combiner).parameters
    num_args = len(params)
    args = []
    for name, i in zip(params.keys(), identity + identity):
        args.append(Var(name, i.dtype))
    res = combiner(*args)
    if not isinstance(res, tuple):
        res = (res,)
    return CommReducer(args[: num_args // 2], args[num_args // 2 :], res, identity)
