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
import inspect
from typing import Union

from . import doc


class Source:
    source_name: str
    start_line: int
    start_column: int
    source: str
    full_source: str

    def __init__(self, program: Union[str, doc.AST]):
        if isinstance(program, str):
            self.source_name = "<str>"
            self.start_line = 1
            self.start_column = 0
            self.source = program
            self.full_source = program
        else:
            self.source_name = inspect.getsourcefile(program)  # type: ignore
            lines, self.start_line = inspect.getsourcelines(program)  # type: ignore
            if lines:
                self.start_column = len(lines[0]) - len(lines[0].lstrip())
            else:
                self.start_column = 0
            if self.start_column and lines:
                self.source = "\n".join([l[self.start_column :].rstrip() for l in lines])
            else:
                self.source = "".join(lines)
            try:
                # It will cause a problem when running in Jupyter Notebook.
                # `mod` will be <module '__main__'>, which is a built-in module
                # and `getsource` will throw a TypeError
                mod = inspect.getmodule(program)
                if mod:
                    self.full_source = inspect.getsource(mod)
                else:
                    self.full_source = self.source
            except TypeError:
                # It's a work around for Jupyter problem.
                # Since `findsource` is an internal API of inspect, we just use it
                # as a fallback method.
                src, _ = inspect.findsource(program)  # type: ignore
                self.full_source = "".join(src)

    def as_ast(self) -> doc.AST:
        return doc.parse(self.source)
