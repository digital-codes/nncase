﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.Transform.Pattern.F.Math;

namespace Nncase.Transform.Pattern
{
    /// <summary>
    /// Math operators for <see cref="ExprPattern"/>.
    /// </summary>
    public partial record ExprPattern
    {
        /// <summary>
        /// Unary neg.
        /// </summary>
        /// <param name="lhs">Source operand.</param>
        /// <returns>Result.</returns>
        public static CallPattern operator -(ExprPattern lhs) => Neg(Utility.GetID(), lhs);

        /// <summary>
        /// Unary bitwise not.
        /// </summary>
        /// <param name="lhs">Source operand.</param>
        /// <returns>Result.</returns>
        public static CallPattern operator ~(ExprPattern lhs) => BitwiseNot(Utility.GetID(), lhs);

        /// <summary>
        /// Unary logical not.
        /// </summary>
        /// <param name="lhs">Source operand.</param>
        /// <returns>Result.</returns>
        public static CallPattern operator !(ExprPattern lhs) => LogicalNot(Utility.GetID(), lhs);

        /// <summary>
        /// Binary add.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result.</returns>
        public static CallPattern operator +(ExprPattern lhs, ExprPattern rhs) => Add(Utility.GetID(), lhs, rhs);

        /// <summary>
        /// Binary sub.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result.</returns>
        public static CallPattern operator -(ExprPattern lhs, ExprPattern rhs) => Sub(Utility.GetID(), lhs, rhs);

        /// <summary>
        /// Binary mul.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result.</returns>
        public static CallPattern operator *(ExprPattern lhs, ExprPattern rhs) => Mul(Utility.GetID(), lhs, rhs);

        /// <summary>
        /// Binary div.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result.</returns>
        public static CallPattern operator /(ExprPattern lhs, ExprPattern rhs) => Div(Utility.GetID(), lhs, rhs);

        /// <summary>
        /// Binary mod.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result.</returns>
        public static CallPattern operator %(ExprPattern lhs, ExprPattern rhs) => Mod(Utility.GetID(), lhs, rhs);

        /// <summary>
        /// Binary bitwise and.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result.</returns>
        public static CallPattern operator &(ExprPattern lhs, ExprPattern rhs) => BitwiseAnd(Utility.GetID(), lhs, rhs);

        /// <summary>
        /// Binary bitwise or.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result.</returns>
        public static CallPattern operator |(ExprPattern lhs, ExprPattern rhs) => BitwiseOr(Utility.GetID(), lhs, rhs);

        /// <summary>
        /// Binary bitwise xor.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result.</returns>
        public static CallPattern operator ^(ExprPattern lhs, ExprPattern rhs) => BitwiseXor(Utility.GetID(), lhs, rhs);
    }
}
