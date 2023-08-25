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
# pylint: disable=invalid-name
"""Default legalization function for ccl operators."""
import logging

from ...op import call_pure_packed
from ...block_builder import BlockBuilder
from ...expr import Call, Expr, ShapeExpr
from .common import register_legalize


@register_legalize("relax.ccl.allreduce")
def _allreduce(bb: BlockBuilder, call: Call) -> Expr:
    op_type_str = call.attrs.op_type
    op_type_map = {
        "sum": 0,
    }

    if op_type_str not in op_type_map:
        logging.info(
            f"Allreduce only supports limited reduction operations, including {op_type_map.keys()}, but got {op_type_str}"
        )
    return call_pure_packed(
        "runtime.disco.allreduce",
        call.args[0],
        ShapeExpr([op_type_map[op_type_str]]),
        sinfo_args=call.args[0].struct_info,
    )
