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
# pylint: disable=missing-docstring, invalid-name
import inspect
import sys
from typing import Union

from . import doc
import re

_findsource = inspect.findsource


def findsource(object):
    if not inspect.isclass(object):
        return _findsource(object)

    file = inspect.getsourcefile(object)
    if file:
        inspect.linecache.checkcache(file)
    else:
        file = inspect.getfile(object)
        if not (file.startswith("<") and file.endswith(">")):
            raise OSError("source code not available")

    module = inspect.getmodule(object, file)
    if module:
        lines = inspect.linecache.getlines(file, module.__dict__)
    else:
        lines = inspect.linecache.getlines(file)
    if not lines:
        raise OSError("could not get source code")
    qual_name = object.__qualname__.replace(".<locals>", "<locals>").split(".")
    pat_list = []
    for qn in qual_name:
        if qn.endswith("<locals>"):
            pat_list.append(re.compile(r"^(\s*)def\s*" + qn[:-8] + r"\b"))
        else:
            pat_list.append(re.compile(r"^(\s*)class\s*" + qn + r"\b"))
    for i in range(len(lines)):
        match = pat_list[0].match(lines[i])
        if match:
            pat_list.pop(0)
        if not pat_list:
            return lines, i
    raise OSError("could not find class definition")


def getsourcelines(object):
    object = inspect.unwrap(object)
    lines, lnum = findsource(object)
    return inspect.getblock(lines[lnum:]), lnum + 1


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
            return

        self.source_name = inspect.getsourcefile(program)  # type: ignore
        lines, self.start_line = getsourcelines(program)  # type: ignore
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


_getfile = inspect.getfile  # pylint: disable=invalid-name


def _patched_inspect_getfile(obj):
    if not inspect.isclass(obj):
        return _getfile(obj)
    mod = getattr(obj, "__module__", None)
    if mod is not None:
        file = getattr(sys.modules[mod], "__file__", None)
        if file is not None:
            return file
    for _, member in inspect.getmembers(obj):
        if inspect.isfunction(member):
            if obj.__qualname__ + "." + member.__name__ == member.__qualname__:
                return inspect.getfile(member)
    raise TypeError(f"Source for {obj:!r} not found")


inspect.getfile = _patched_inspect_getfile
