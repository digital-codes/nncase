/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

template result<void> reference::equal<uint8_t>(const uint8_t *input_a, const uint8_t *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_strides) noexcept;

template result<void> reference::equal<float>(const float *input_a, const float *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_strides) noexcept;

template result<void> reference::equal<int64_t>(const int64_t *input_a, const int64_t *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_strides) noexcept;

template <typename T>
result<void> reference::equal(const T *input_a, const T *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_strides) noexcept
{
    const auto out_shape = kernels::detail::get_binary_output_shape(in_a_shape, in_b_shape);
    return apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        const auto in_a_index = kernels::detail::get_reduced_offset(index, in_a_shape);
        const auto in_b_index = kernels::detail::get_reduced_offset(index, in_b_shape);

        const auto a = input_a[offset(in_a_strides, in_a_index)];
        const auto b = input_b[offset(in_b_strides, in_b_index)];

        output[offset(out_strides, index)] = (a == b);
        return ok();
    });
}