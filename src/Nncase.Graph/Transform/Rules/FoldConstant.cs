﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Pattern;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.F.NN;
using static Nncase.Pattern.Utility;
using Nncase.IR;
using Nncase.Evaluator;

namespace Nncase.Transform.DataFlow.Rules
{
    public class FoldConstCall : PatternRule
    {
        public FoldConstCall()
        {
            Pattern = IsCall(IsWildCard(), IsVArgsRepeat((n, pats) =>
             {
                 foreach (var i in Enumerable.Range(0, n))
                 {
                     pats.Add(IsConst());
                 }
             }));
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            var expr = result[Pattern];
            return Evaluator.Evaluator.Eval(expr).ToConst();
        }
    }

    public class FoldShapeOp : PatternRule
    {
        WildCardPattern wc = "input";

        public FoldShapeOp()
        {
            Pattern = ShapeOp(wc);
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            return Const.FromShape(result[wc].CheckedShape);
        }
    }
}