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
"""Hybrid Script Parser Special Stmt Functions

This module provides the functions registered into parser under special_stmt category.
Special Stmt functions are used to provide some primitive functions for specific use.
Typically, a special stmt function has return value and accepts parser and
node as its first 2 arguments.
"""
# pylint: disable=unused-argument
import tvm.tir
from tvm.tir import ir_pass as _pass


def buffer_bind(parser, node, var, shape, dtype="float32"):
    """ Special function buffer_bind(var, shape, dtype)

    Example
    -------
    .. code-block:: python

        A = buffer_bind(a, (128, 128), dtype="float32")

    """

    if var not in parser.params:
        parser.report_error("Can not bind non-input args to buffer")
    return tvm.tir.decl_buffer(shape, dtype=dtype, name=parser._assign_target)


def buffer_allocate(parser, node, shape, dtype="float32", scope=""):
    """ Special function buffer_allocate(var, shape, dtype, scope)

    Example
    -------
    .. code-block:: python

        A = buffer_allocate((128, 128), dtype="float32")

    """
    _buffer = tvm.tir.decl_buffer(shape, dtype=dtype, name=parser._assign_target)
    parser.scope_emitter.alloc(tvm.tir.BufferAllocate(_buffer, scope))
    return _buffer


def block_vars(parser, node, begin, end, name="bv", iter_type="data_par"):
    """ Special function buffer_bind(var, shape, dtype, name)

    Example
    -------
    .. code-block:: python

        vi(0, 128, iter_type="reduce")

    """
    extent = end if begin == 0 else _pass.Simplify(end - begin)
    block_var_dom = tvm.ir.Range.make_by_min_extent(begin, extent)

    if iter_type == "data_par":
        iter_type_id = 0
    elif iter_type == "reduce":
        iter_type_id = 2
    elif iter_type == "scan":
        iter_type_id = 3
    elif iter_type == "opaque":
        iter_type_id = 4
    else:
        raise ValueError("Unknown iter_type")
    block_var = tvm.tir.IterVar(block_var_dom, name, iter_type_id)
    return block_var
