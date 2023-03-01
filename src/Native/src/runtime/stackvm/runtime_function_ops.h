/* This file is generated by tools/stackvm_gen/IsaGen at 02/24/2023 15:52:41
 * +08:00.
 *
 * Copyright 2019-2021 Canaan Inc.
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

result<void> visit(const nop_op_t &op) noexcept override;
result<void> visit(const br_op_t &op) noexcept override;
result<void> visit(const br_true_op_t &op) noexcept override;
result<void> visit(const br_false_op_t &op) noexcept override;
result<void> visit(const ret_op_t &op) noexcept override;
result<void> visit(const call_op_t &op) noexcept override;
result<void> visit(const ecall_op_t &op) noexcept override;
result<void> visit(const extcall_op_t &op) noexcept override;
result<void> visit(const cuscall_op_t &op) noexcept override;
result<void> visit(const throw_op_t &op) noexcept override;
result<void> visit(const break_op_t &op) noexcept override;
result<void> visit(const ldc_i4_op_t &op) noexcept override;
result<void> visit(const ldnull_op_t &op) noexcept override;
result<void> visit(const ldc_i4_0_op_t &op) noexcept override;
result<void> visit(const ldc_i4_1_op_t &op) noexcept override;
result<void> visit(const ldc_r4_op_t &op) noexcept override;
result<void> visit(const ldind_i1_op_t &op) noexcept override;
result<void> visit(const ldind_i2_op_t &op) noexcept override;
result<void> visit(const ldind_i4_op_t &op) noexcept override;
result<void> visit(const ldind_i_op_t &op) noexcept override;
result<void> visit(const ldind_u1_op_t &op) noexcept override;
result<void> visit(const ldind_u2_op_t &op) noexcept override;
result<void> visit(const ldind_u4_op_t &op) noexcept override;
result<void> visit(const ldind_u_op_t &op) noexcept override;
result<void> visit(const ldind_br2_op_t &op) noexcept override;
result<void> visit(const ldind_r4_op_t &op) noexcept override;
result<void> visit(const stind_i1_op_t &op) noexcept override;
result<void> visit(const stind_i2_op_t &op) noexcept override;
result<void> visit(const stind_i4_op_t &op) noexcept override;
result<void> visit(const stind_i_op_t &op) noexcept override;
result<void> visit(const stind_br2_op_t &op) noexcept override;
result<void> visit(const stind_r4_op_t &op) noexcept override;
result<void> visit(const lea_gp_op_t &op) noexcept override;
result<void> visit(const ldelem_i1_op_t &op) noexcept override;
result<void> visit(const ldelem_i2_op_t &op) noexcept override;
result<void> visit(const ldelem_i4_op_t &op) noexcept override;
result<void> visit(const ldelem_i_op_t &op) noexcept override;
result<void> visit(const ldelem_u1_op_t &op) noexcept override;
result<void> visit(const ldelem_u2_op_t &op) noexcept override;
result<void> visit(const ldelem_u4_op_t &op) noexcept override;
result<void> visit(const ldelem_u_op_t &op) noexcept override;
result<void> visit(const ldelem_br2_op_t &op) noexcept override;
result<void> visit(const ldelem_r4_op_t &op) noexcept override;
result<void> visit(const stelem_i1_op_t &op) noexcept override;
result<void> visit(const stelem_i2_op_t &op) noexcept override;
result<void> visit(const stelem_i4_op_t &op) noexcept override;
result<void> visit(const stelem_i_op_t &op) noexcept override;
result<void> visit(const stelem_br2_op_t &op) noexcept override;
result<void> visit(const stelem_r4_op_t &op) noexcept override;
result<void> visit(const ldarg_op_t &op) noexcept override;
result<void> visit(const ldarg_0_op_t &op) noexcept override;
result<void> visit(const ldarg_1_op_t &op) noexcept override;
result<void> visit(const ldarg_2_op_t &op) noexcept override;
result<void> visit(const ldarg_3_op_t &op) noexcept override;
result<void> visit(const ldarg_4_op_t &op) noexcept override;
result<void> visit(const ldarg_5_op_t &op) noexcept override;
result<void> visit(const ldshape_op_t &op) noexcept override;
result<void> visit(const ldstrides_op_t &op) noexcept override;
result<void> visit(const ldtuple_elem_op_t &op) noexcept override;
result<void> visit(const ldtuple_op_t &op) noexcept override;
result<void> visit(const lddatatype_op_t &op) noexcept override;
result<void> visit(const ldtensor_op_t &op) noexcept override;
result<void> visit(const ldlocal_op_t &op) noexcept override;
result<void> visit(const stlocal_op_t &op) noexcept override;
result<void> visit(const ldscalar_op_t &op) noexcept override;
result<void> visit(const dup_op_t &op) noexcept override;
result<void> visit(const pop_op_t &op) noexcept override;
result<void> visit(const neg_op_t &op) noexcept override;
result<void> visit(const add_op_t &op) noexcept override;
result<void> visit(const sub_op_t &op) noexcept override;
result<void> visit(const mul_op_t &op) noexcept override;
result<void> visit(const div_op_t &op) noexcept override;
result<void> visit(const div_u_op_t &op) noexcept override;
result<void> visit(const rem_op_t &op) noexcept override;
result<void> visit(const rem_u_op_t &op) noexcept override;
result<void> visit(const and_op_t &op) noexcept override;
result<void> visit(const or_op_t &op) noexcept override;
result<void> visit(const xor_op_t &op) noexcept override;
result<void> visit(const not_op_t &op) noexcept override;
result<void> visit(const shl_op_t &op) noexcept override;
result<void> visit(const shr_op_t &op) noexcept override;
result<void> visit(const shr_u_op_t &op) noexcept override;
result<void> visit(const clt_op_t &op) noexcept override;
result<void> visit(const clt_u_op_t &op) noexcept override;
result<void> visit(const cle_op_t &op) noexcept override;
result<void> visit(const cle_u_op_t &op) noexcept override;
result<void> visit(const ceq_op_t &op) noexcept override;
result<void> visit(const cge_op_t &op) noexcept override;
result<void> visit(const cge_u_op_t &op) noexcept override;
result<void> visit(const cgt_op_t &op) noexcept override;
result<void> visit(const cgt_u_op_t &op) noexcept override;
result<void> visit(const cne_op_t &op) noexcept override;
result<void> visit(const conv_i1_op_t &op) noexcept override;
result<void> visit(const conv_i2_op_t &op) noexcept override;
result<void> visit(const conv_i4_op_t &op) noexcept override;
result<void> visit(const conv_i_op_t &op) noexcept override;
result<void> visit(const conv_u1_op_t &op) noexcept override;
result<void> visit(const conv_u2_op_t &op) noexcept override;
result<void> visit(const conv_u4_op_t &op) noexcept override;
result<void> visit(const conv_u_op_t &op) noexcept override;
result<void> visit(const conv_br2_op_t &op) noexcept override;
result<void> visit(const conv_r4_op_t &op) noexcept override;

result<void> visit(const tensor_batch_normalization_op_t &op) noexcept override;
result<void> visit(const tensor_batch_to_space_op_t &op) noexcept override;
result<void> visit(const tensor_binary_op_t &op) noexcept override;
result<void> visit(const tensor_bitcast_op_t &op) noexcept override;
result<void> visit(const tensor_broadcast_op_t &op) noexcept override;
result<void> visit(const tensor_cast_op_t &op) noexcept override;
result<void> visit(const tensor_celu_op_t &op) noexcept override;
result<void> visit(const tensor_clamp_op_t &op) noexcept override;
result<void> visit(const tensor_compare_op_t &op) noexcept override;
result<void> visit(const tensor_concat_op_t &op) noexcept override;
result<void> visit(const tensor_condition_op_t &op) noexcept override;
result<void> visit(const tensor_constant_of_shape_op_t &op) noexcept override;
result<void> visit(const tensor_conv2d_op_t &op) noexcept override;
result<void> visit(const tensor_conv2d_transpose_op_t &op) noexcept override;
result<void> visit(const tensor_cum_sum_op_t &op) noexcept override;
result<void> visit(const tensor_dequantize_op_t &op) noexcept override;
result<void> visit(const tensor_elu_op_t &op) noexcept override;
result<void> visit(const tensor_erf_op_t &op) noexcept override;
result<void> visit(const tensor_expand_op_t &op) noexcept override;
result<void> visit(const tensor_fake_dequantize_op_t &op) noexcept override;
result<void> visit(const tensor_fake_quantize_op_t &op) noexcept override;
result<void> visit(const tensor_flatten_op_t &op) noexcept override;
result<void> visit(const tensor_gather_op_t &op) noexcept override;
result<void> visit(const tensor_gather_nd_op_t &op) noexcept override;
result<void> visit(const tensor_get_item_op_t &op) noexcept override;
result<void> visit(const tensor_hard_sigmoid_op_t &op) noexcept override;
result<void> visit(const tensor_hard_swish_op_t &op) noexcept override;
result<void> visit(const tensor_hardmax_op_t &op) noexcept override;
result<void>
visit(const tensor_instance_normalization_op_t &op) noexcept override;
result<void> visit(const tensor_l2_normalization_op_t &op) noexcept override;
result<void> visit(const tensor_leaky_relu_op_t &op) noexcept override;
result<void> visit(const tensor_log_softmax_op_t &op) noexcept override;
result<void> visit(const tensor_lp_normalization_op_t &op) noexcept override;
result<void> visit(const tensor_lrn_op_t &op) noexcept override;
result<void> visit(const tensor_lstm_op_t &op) noexcept override;
result<void> visit(const tensor_mat_mul_op_t &op) noexcept override;
result<void> visit(const tensor_normal_op_t &op) noexcept override;
result<void> visit(const tensor_normal_like_op_t &op) noexcept override;
result<void> visit(const tensor_one_hot_op_t &op) noexcept override;
result<void> visit(const tensor_pad_op_t &op) noexcept override;
result<void> visit(const tensor_prelu_op_t &op) noexcept override;
result<void> visit(const tensor_prod_op_t &op) noexcept override;
result<void> visit(const tensor_quant_param_of_op_t &op) noexcept override;
result<void> visit(const tensor_quantize_op_t &op) noexcept override;
result<void> visit(const tensor_range_op_t &op) noexcept override;
result<void> visit(const tensor_range_of_op_t &op) noexcept override;
result<void> visit(const tensor_reduce_op_t &op) noexcept override;
result<void> visit(const tensor_reduce_arg_op_t &op) noexcept override;
result<void> visit(const tensor_reduce_window2d_op_t &op) noexcept override;
result<void> visit(const tensor_relu_op_t &op) noexcept override;
result<void> visit(const tensor_relu6_op_t &op) noexcept override;
result<void> visit(const tensor_require_op_t &op) noexcept override;
result<void> visit(const tensor_reshape_op_t &op) noexcept override;
result<void> visit(const tensor_resize_image_op_t &op) noexcept override;
result<void> visit(const tensor_reverse_sequence_op_t &op) noexcept override;
result<void> visit(const tensor_select_op_t &op) noexcept override;
result<void> visit(const tensor_selu_op_t &op) noexcept override;
result<void> visit(const tensor_shape_of_op_t &op) noexcept override;
result<void> visit(const tensor_sigmoid_op_t &op) noexcept override;
result<void> visit(const tensor_size_of_op_t &op) noexcept override;
result<void> visit(const tensor_slice_op_t &op) noexcept override;
result<void> visit(const tensor_softmax_op_t &op) noexcept override;
result<void> visit(const tensor_softplus_op_t &op) noexcept override;
result<void> visit(const tensor_softsign_op_t &op) noexcept override;
result<void> visit(const tensor_space_to_batch_op_t &op) noexcept override;
result<void> visit(const tensor_split_op_t &op) noexcept override;
result<void> visit(const tensor_squeeze_op_t &op) noexcept override;
result<void> visit(const tensor_stack_op_t &op) noexcept override;
result<void> visit(const tensor_tile_op_t &op) noexcept override;
result<void> visit(const tensor_top_k_op_t &op) noexcept override;
result<void> visit(const tensor_transpose_op_t &op) noexcept override;
result<void> visit(const tensor_unary_op_t &op) noexcept override;
result<void> visit(const tensor_uniform_op_t &op) noexcept override;
result<void> visit(const tensor_uniform_like_op_t &op) noexcept override;
result<void> visit(const tensor_unsqueeze_op_t &op) noexcept override;
result<void> visit(const tensor_where_op_t &op) noexcept override;
