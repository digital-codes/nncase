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
#include <nncase/kernels/cpu/reference/runtime_types.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/util.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu::reference;
using namespace nncase::kernels::stackvm;

namespace {
template <class T, class TOp>
result<void> unary_impl(TOp &&op, const T *input, T *output,
                         [[maybe_unused]] const dims_t &input_shape, const strides_t &input_strides,
                         const dims_t &out_shape, const strides_t &out_strides,
                         NNCASE_UNUSED kernel_context &context) noexcept {
    return apply(out_shape, [&](const dims_t &index) -> result<void> {
        const auto v = input[offset(input_strides, index)];
        output[offset(out_strides, index)] = op(v);
        return ok();
    });
}

#define UNARY_IMPL_OP(op, funct)                                              \
    case unary_op_t::op:                                                      \
        return unary_impl(funct, input, output, input_shape, input_strides,    \
                           out_shape, out_strides,     \
                           context)

template <class T>
result<void> unary_impl(unary_op_t op, const T *input, T *output,
                         const dims_t &input_shape, const strides_t &input_strides,
                         const dims_t &out_shape, const strides_t &out_strides,
                         NNCASE_UNUSED kernel_context &context) noexcept {
    switch (op) {
        UNARY_IMPL_OP(abs, fabsf);
        UNARY_IMPL_OP(acos, acosf);
        UNARY_IMPL_OP(asin, asinf);
        UNARY_IMPL_OP(ceil, ceilf);
        UNARY_IMPL_OP(cos, cosf);
        UNARY_IMPL_OP(exp, expf);
        UNARY_IMPL_OP(floor, floorf);
        UNARY_IMPL_OP(log, logf);
        UNARY_IMPL_OP(logical_not, [](float v) { return !v; });
        UNARY_IMPL_OP(neg, std::negate<float>());
        UNARY_IMPL_OP(round, roundf);
        UNARY_IMPL_OP(rsqrt, [](float v) { return 1.f / sqrtf(v); });
        UNARY_IMPL_OP(sign, [](float v) { return (0.f < v) - (v < 0.f); });
        UNARY_IMPL_OP(sin, sinf);
        UNARY_IMPL_OP(sqrt, sqrtf);
        UNARY_IMPL_OP(square, [](float v) { return v * v; });
        UNARY_IMPL_OP(tanh, tanhf);
    default:
        return err(std::errc::not_supported);
    }
}

#define UNARY_IMPL_DTYPE(dtype, type)                                         \
    case dtype:                                                                \
        return unary_impl(op, reinterpret_cast<const type *>(input),            \
                           reinterpret_cast<type *>(output), input_shape,        \
                           input_strides, out_shape,     \
                           out_strides, context);

result<void> unary_impl(typecode_t dtype, unary_op_t op, const gsl::byte *input,
                         gsl::byte *output,
                         const dims_t &input_shape, const strides_t &input_strides,
                         const dims_t &out_shape, const strides_t &out_strides,
                         kernel_context &context) noexcept {
    switch (dtype) {
        UNARY_IMPL_DTYPE(dt_float32, float)
    default:
        return err(nncase_errc::datatype_mismatch);
    }
}
} // namespace

result<value_t> kernels::stackvm::unary(unary_op_t unary_op, value_t input,
                                       value_t output,
                                        kernel_context &context) {
    try_input(input_mem, input);
    try_var(typoecode, to_typecode(input_tensor->dtype()));
    auto dtype = input_tensor->dtype();
    try_output(out_mem, output, dtype, input_tensor->shape());

    try_(unary_impl(typoecode, unary_op, input_mem,
                        out_mem,
                        input_tensor->shape(), input_tensor->strides(),
                        output_tensor->shape(), output_tensor->strides(), context));
    return ok(output);
}