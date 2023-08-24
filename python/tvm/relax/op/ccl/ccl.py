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
"""Relax Collective Communications Library (CCL) operators"""
from . import _ffi_api


def allreduce(x, op_type: str = "sum"):
    """
    TODO
    """
    op_type_map = {"sum": 0, "prod": 1, "max": 2, "min": 3}
    assert op_type in op_type_map, "unsupported all reduction operations."
    return _ffi_api.allreduce(x, op_type_map[op_type])
