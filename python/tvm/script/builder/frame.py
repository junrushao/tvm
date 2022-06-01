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
"""TVM Script Frames"""
from tvm._ffi import register_object as _register_object

from tvm.runtime import Object

from . import _ffi_api


@_register_object("script.builder.Frame")
class Frame(Object):
    def __enter__(self) -> "Frame":
        _ffi_api.FrameEnter(self)
        return self

    def __exit__(self, ptype, value, trace) -> None:
        _ffi_api.FrameExit(self)


@_register_object("script.builder.IRModuleFrame")
class IRModuleFrame(Frame):
    def __init__(self) -> None:
        self.__init_handle_by_constructor__(_ffi_api.IRModuleFrame)

    def __enter__(self) -> "IRModuleFrame":
        _ffi_api.FrameEnter(self)
        return self

    def __exit__(self, ptype, value, trace) -> None:
        _ffi_api.FrameExit(self)
