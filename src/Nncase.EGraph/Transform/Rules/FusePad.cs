using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.Transform.Pattern.NN;
using Nncase.Transform.Pattern.Tensors;
using static Nncase.Transform.Pattern.Utility;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using System.Numerics.Tensors;
using Nncase.IR;

namespace Nncase.Transform.Rule
{
    public class FoldPadConv2d : EGraphRule
    {
        Conv2DWrapper conv2d;
        PadWrapper pad;
        public FoldPadConv2d()
        {
            pad = IsPad(IsWildCard(), IsConst(), PadMode.Constant, IsConst());
            Pattern = conv2d = IsConv2D(pad, PadMode.Constant);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            pad.Bind(result);
            conv2d.Bind(result);
            var pads = pad.Pads<Const>().ToTensor<int>();
            var padv = pad.Value<Const>().ToScalar<float>();
            if (pads.Dimensions[0] == 4
            && pads[2, 0] >= 0 && pads[2, 1] >= 0
            && pads[3, 0] >= 0 && pads[3, 1] >= 0
            && ((pads[2, 0] + pads[2, 1]) > 0 || (pads[3, 0] + pads[3, 1]) > 0)
            && padv == .0f)
            {
                var newpads = new DenseTensor<int>(new[] { 2, 2 });
                for (int i = 2; i < 4; i++)
                {
                    if (pads[i, 0] > 0)
                    {
                        newpads[i - 2, 0] += pads[i, 0];
                        pads[i, 0] = 0;
                    }

                    if (pads[i, 1] > 0)
                    {
                        newpads[i - 2, 1] += pads[i, 1];
                        pads[i, 1] = 0;
                    }
                }
                return Conv2D(Pad(pad.Input(), Const.FromTensor(pads), pad.PadMode, pad.Value()), conv2d.Weights(), conv2d.Bias(), Const.FromTensor(newpads), conv2d.Stride(), conv2d.Dilation(), conv2d.PadMode, conv2d.Groups());
            }
            return null;
        }

    }
}