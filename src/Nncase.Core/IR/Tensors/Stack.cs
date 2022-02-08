// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.Utility;

namespace Nncase.IR.Tensors;

/// <summary>
/// Stack expression.
/// </summary>
public sealed record Stack() : Op
{
    /// <summary>
    /// Gets inputs.
    /// </summary>
    public static readonly ParameterInfo Inputs = new(typeof(Stack), 0, "inputs");

    /// <summary>
    /// Gets axis.
    /// </summary>
    public static readonly ParameterInfo Axis = new(typeof(Stack), 1, "axis", IsScalar() & IsIntegral());
}
