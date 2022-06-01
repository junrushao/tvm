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
from tvm._ffi import register_object as _register_object

from tvm.runtime import Object

from . import _ffi_api
from .base import TIRFrame
from typing import List

@_register_object("script.builder.tir.ForFrame")
class ForFrame(TIRFrame):
    def __enter__(self) -> List[Object]:
        _ffi_api.ForFrameEnter(self)
        return self.vars

    def __exit__(self, ptype, value, trace) -> None:
        _ffi_api.ForFrameExit(self)


def Serial(min_val, extent, attrs) -> Object:
    return _ffi_api.Serial(min_val, extent, attrs)

def Parallel(min_val, extent, attrs) -> Object:
    return _ffi_api.Parallel(min_val, extent, attrs)

def Vectorized(min_val, extent, attrs) -> Object:
    return _ffi_api.Vectorized(min_val, extent, attrs)

def Unroll(min_val, extent, attrs) -> Object:
    return _ffi_api.Unroll(min_val, extent, attrs)

def ThreadBinding(min_val, extent, attrs) -> Object:
    return _ffi_api.ThreadBinding(min_val, extent, attrs)

def Grid(*extents) -> Object:
    return _ffi_api.Grid(list(extents))
