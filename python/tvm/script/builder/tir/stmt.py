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
"""TVM Script TIR For Frame"""
from typing import List

from tvm._ffi import register_object as _register_object
from tvm.tir import Var

from . import _ffi_api
from .. import _ffi_api as _base_ffi_api
from .base import TIRFrame


@_register_object("script.builder.tir.AssertFrame")
class AssertFrame(TIRFrame):
    ...


@_register_object("script.builder.tir.LetFrame")
class LetFrame(TIRFrame):
    ...


def Assert(condition, message) -> AssertFrame:
    return _ffi_api.AssertFrame(condition, message)  # pylint: disable=no-member # type: ignore


def Let(var, value) -> LetFrame:
    return _ffi_api.LetFrame(var, value)  # pylint: disable=no-member # type: ignore
