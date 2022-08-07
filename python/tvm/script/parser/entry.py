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
# pylint: disable=missing-docstring
"""The entry point of TVM parser."""
from typing import Any, Union

from tvm.ir.ir_builder import IRBuilder

from . import doc
from .parser import Parser
from .source import Source


def parse(program: Union[doc.AST, Any, str], extra_vars=None):
    if isinstance(program, str) and extra_vars is None:
        from tvm.script.parser import ir # pylint: disable=import-outside-toplevel
        from tvm.script.parser import tir  # pylint: disable=import-outside-toplevel

        extra_vars = {
            "I": ir,
            "ir": ir,
            "T": tir,
            "tir": tir,
        }
    source = Source(program)
    parser = Parser(source)
    with IRBuilder() as builder:
        parser.parse(extra_vars=extra_vars)
    return builder.get()
