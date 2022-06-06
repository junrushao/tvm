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
"""The symbol table of variable values"""

from collections import defaultdict
from typing import Any, Callable, Dict, List

from .utils import deferred


class VarTableFrame:
    vars: List[str]

    def __init__(self):
        self.vars = []

    def add(self, var: str):
        self.vars.append(var)

    def pop_all(self, fn_pop: Callable[[str], None]):
        for var in self.vars:
            fn_pop(var)
        self.vars = []


class VarTable:

    frames: List[VarTableFrame]
    name2value: Dict[str, List[Any]]

    def __init__(self):
        self.frames = []
        self.name2value = defaultdict(list)

    def with_frame(self):
        self.frames.append(VarTableFrame())

        def pop_frame():
            frame = self.frames.pop()
            frame.pop_all(lambda var: self.name2value[var].pop())

        return deferred(pop_frame)

    def add(self, var: str, value: Any):
        self.frames[-1].add(var)
        self.name2value[var].append(value)
