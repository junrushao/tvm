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
"""The core parser"""
import ast
from typing import Any, Dict, List, Optional, Union

from . import dispatch
from .evaluator import eval_assign, eval_expr
from .var_table import VarTable


class Parser(ast.NodeVisitor):
    """The TVMScript parser"""

    dispatch_tokens: List[str]
    var_table: VarTable

    def __init__(self) -> None:
        self.dispatch_tokens = ["default"]
        self.var_table = VarTable()

    def _dispatch(self, type_name: str) -> dispatch.ParseMethod:
        for token in [self.dispatch_tokens[-1], "default"]:
            result = dispatch.get(token=token, type_name=type_name, default=None)
            if result is not None:
                return result
        return self.generic_visit

    def eval_expr(
        self,
        node: Union[ast.Expression, ast.expr],
        extra_vars: Optional[Dict[str, Any]] = None,
    ) -> Any:
        var_values = self.var_table.get()
        if extra_vars is not None:
            for k, v in extra_vars.items():
                var_values[k] = v
        return eval_expr(node, var_values)

    def eval_assign(
        self,
        target: ast.expr,
        source: Any,
    ):
        var_values = eval_assign(target, source)
        for k, v in var_values.items():
            self.var_table.add(k, v)

    def visit_arg(self, node: ast.arg) -> Any:
        self._dispatch("arg")(self, node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # pylint: disable=invalid-name
        self._dispatch("FunctionDef")(self, node)

    def visit_For(self, node: ast.For) -> Any:  # pylint: disable=invalid-name
        self._dispatch("For")(self, node)

    def visit_With(self, node: ast.With) -> Any:  # pylint: disable=invalid-name
        self._dispatch("With")(self, node)

    def visit_Assign(self, node: ast.Assign) -> Any:  # pylint: disable=invalid-name
        self._dispatch("Assign")(self, node)


@dispatch.register(token="default", type_name="FunctionDef")
def visit_function_def(self: Parser, node: ast.FunctionDef) -> Any:
    """Visit a function definition"""
    return self.generic_visit(node)
