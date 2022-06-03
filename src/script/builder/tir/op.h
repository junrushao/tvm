/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_SCRIPT_BUILDER_TIR_OP_H_
#define TVM_SCRIPT_BUILDER_TIR_OP_H_

#include <tvm/tir/op.h>

#include "../builder.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

using tvm::abs;
using tvm::add;
using tvm::all;
using tvm::any;
using tvm::bitwise_and;
using tvm::bitwise_neg;
using tvm::bitwise_or;
using tvm::bitwise_xor;
using tvm::cast;
using tvm::ceil;
using tvm::ceildiv;
using tvm::div;
using tvm::equal;
using tvm::floor;
using tvm::floordiv;
using tvm::floormod;
using tvm::greater;
using tvm::greater_equal;
using tvm::if_then_else;
using tvm::indexdiv;
using tvm::indexmod;
using tvm::infinity;
using tvm::isfinite;
using tvm::isinf;
using tvm::isnan;
using tvm::LargeUIntImm;
using tvm::left_shift;
using tvm::less;
using tvm::less_equal;
using tvm::likely;
using tvm::logical_and;
using tvm::logical_not;
using tvm::logical_or;
using tvm::max;
using tvm::max_value;
using tvm::min;
using tvm::min_value;
using tvm::mul;
using tvm::nearbyint;
using tvm::neg;
using tvm::not_equal;
using tvm::pow;
using tvm::prod;
using tvm::q_multiply_shift;
using tvm::right_shift;
using tvm::round;
using tvm::shapediv;
using tvm::sum;
using tvm::trunc;
using tvm::truncdiv;
using tvm::truncmod;

using tvm::acos;
using tvm::acosh;
using tvm::asin;
using tvm::asinh;
using tvm::atan;
using tvm::atanh;
using tvm::clz;
using tvm::cos;
using tvm::cosh;
using tvm::erf;
using tvm::exp;
using tvm::exp10;
using tvm::exp2;
using tvm::log;
using tvm::log10;
using tvm::log2;
using tvm::popcount;
using tvm::rsqrt;
using tvm::sigmoid;
using tvm::sin;
using tvm::sinh;
using tvm::sqrt;
using tvm::tan;
using tvm::tanh;

using tvm::atan2;
using tvm::copysign;
using tvm::hypot;
using tvm::ldexp;
using tvm::nextafter;

using tvm::infinity;

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_TIR_OP_H_
