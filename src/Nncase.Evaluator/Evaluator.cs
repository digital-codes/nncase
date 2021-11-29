﻿using Nncase.Evaluator.Ops;
using Nncase.IR;
using TorchSharp;

namespace Nncase.Evaluator
{
    public static class Evaluator
    {
        public static torch.Tensor Eval(this Expr expr)
        {
            var evaluatorVisitor = new EvaluatorVisitor();
            return evaluatorVisitor.Visit(expr);
            // todo:torch.Tensor to nncase Tensor
        }
    }
}