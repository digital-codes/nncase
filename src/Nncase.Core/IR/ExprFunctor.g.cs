﻿
//---------------------------------------------------------------------------------------------------
// <auto-generated>
//    This code was generated by T4 template.
//    Changes to this file may cause incorrect behavior and will be lost if the code is regenerated.
// </auto-generated>
//---------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Reactive;

namespace Nncase.IR;

public partial class ExprFunctor<TExprResult, TTypeResult, TContext>
{
    /// <summary>
    /// Visit <see cref="BaseFunction"/>.
    /// </summary>
    internal protected virtual TExprResult VisitBaseFunction(BaseFunction expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="Call"/>.
    /// </summary>
    internal protected virtual TExprResult VisitCall(Call expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="Const"/>.
    /// </summary>
    internal protected virtual TExprResult VisitConst(Const expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="Function"/>.
    /// </summary>
    internal protected virtual TExprResult VisitFunction(Function expr, TContext context) => VisitBaseFunction(expr, context);

    /// <summary>
    /// Visit <see cref="Fusion"/>.
    /// </summary>
    internal protected virtual TExprResult VisitFusion(Fusion expr, TContext context) => VisitBaseFunction(expr, context);

    /// <summary>
    /// Visit <see cref="If"/>.
    /// </summary>
    internal protected virtual TExprResult VisitIf(If expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="Marker"/>.
    /// </summary>
    internal protected virtual TExprResult VisitMarker(Marker expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="None"/>.
    /// </summary>
    internal protected virtual TExprResult VisitNone(None expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="Op"/>.
    /// </summary>
    internal protected virtual TExprResult VisitOp(Op expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="PrimFunctionWrapper"/>.
    /// </summary>
    internal protected virtual TExprResult VisitPrimFunctionWrapper(PrimFunctionWrapper expr, TContext context) => VisitBaseFunction(expr, context);

    /// <summary>
    /// Visit <see cref="TensorConst"/>.
    /// </summary>
    internal protected virtual TExprResult VisitTensorConst(TensorConst expr, TContext context) => VisitConst(expr, context);

    /// <summary>
    /// Visit <see cref="IR.Tuple"/>.
    /// </summary>
    internal protected virtual TExprResult VisitTuple(IR.Tuple expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="TupleConst"/>.
    /// </summary>
    internal protected virtual TExprResult VisitTupleConst(TupleConst expr, TContext context) => VisitConst(expr, context);

    /// <summary>
    /// Visit <see cref="Var"/>.
    /// </summary>
    internal protected virtual TExprResult VisitVar(Var expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.Block"/>.
    /// </summary>
    internal protected virtual TExprResult VisitBlock(TIR.Block expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.Buffer"/>.
    /// </summary>
    internal protected virtual TExprResult VisitBuffer(TIR.Buffer expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.LogicalBuffer"/>.
    /// </summary>
    internal protected virtual TExprResult VisitLogicalBuffer(TIR.LogicalBuffer expr, TContext context) => VisitBuffer(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.PhysicalBuffer"/>.
    /// </summary>
    internal protected virtual TExprResult VisitPhysicalBuffer(TIR.PhysicalBuffer expr, TContext context) => VisitBuffer(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.BufferLoad"/>.
    /// </summary>
    internal protected virtual TExprResult VisitBufferLoad(TIR.BufferLoad expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.BufferRegion"/>.
    /// </summary>
    internal protected virtual TExprResult VisitBufferRegion(TIR.BufferRegion expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.BufferStore"/>.
    /// </summary>
    internal protected virtual TExprResult VisitBufferStore(TIR.BufferStore expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.For"/>.
    /// </summary>
    internal protected virtual TExprResult VisitFor(TIR.For expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.IfThenElse"/>.
    /// </summary>
    internal protected virtual TExprResult VisitIfThenElse(TIR.IfThenElse expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.Let"/>.
    /// </summary>
    internal protected virtual TExprResult VisitLet(TIR.Let expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.PrimFunction"/>.
    /// </summary>
    internal protected virtual TExprResult VisitPrimFunction(TIR.PrimFunction expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.Sequential"/>.
    /// </summary>
    internal protected virtual TExprResult VisitSequential(TIR.Sequential expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.Range"/>.
    /// </summary>
    internal protected virtual TExprResult VisitRange(TIR.Range expr, TContext context) => DefaultVisit(expr, context);

    /// <summary>
    /// Visit <see cref="TIR.IterVar"/>.
    /// </summary>
    internal protected virtual TExprResult VisitIterVar(TIR.IterVar expr, TContext context) => DefaultVisit(expr, context);

}

public partial class ExprFunctor<TExprResult, TTypeResult>
{
    /// <summary>
    /// Visit <see cref="BaseFunction"/>.
    /// </summary>
    internal protected virtual TExprResult VisitBaseFunction(BaseFunction expr) => base.VisitBaseFunction(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitBaseFunction(BaseFunction expr, Unit context) => VisitBaseFunction(expr);
    /// <summary>
    /// Visit <see cref="Call"/>.
    /// </summary>
    internal protected virtual TExprResult VisitCall(Call expr) => base.VisitCall(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitCall(Call expr, Unit context) => VisitCall(expr);
    /// <summary>
    /// Visit <see cref="Const"/>.
    /// </summary>
    internal protected virtual TExprResult VisitConst(Const expr) => base.VisitConst(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitConst(Const expr, Unit context) => VisitConst(expr);
    /// <summary>
    /// Visit <see cref="Function"/>.
    /// </summary>
    internal protected virtual TExprResult VisitFunction(Function expr) => base.VisitFunction(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitFunction(Function expr, Unit context) => VisitFunction(expr);
    /// <summary>
    /// Visit <see cref="Fusion"/>.
    /// </summary>
    internal protected virtual TExprResult VisitFusion(Fusion expr) => base.VisitFusion(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitFusion(Fusion expr, Unit context) => VisitFusion(expr);
    /// <summary>
    /// Visit <see cref="If"/>.
    /// </summary>
    internal protected virtual TExprResult VisitIf(If expr) => base.VisitIf(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitIf(If expr, Unit context) => VisitIf(expr);
    /// <summary>
    /// Visit <see cref="Marker"/>.
    /// </summary>
    internal protected virtual TExprResult VisitMarker(Marker expr) => base.VisitMarker(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitMarker(Marker expr, Unit context) => VisitMarker(expr);
    /// <summary>
    /// Visit <see cref="None"/>.
    /// </summary>
    internal protected virtual TExprResult VisitNone(None expr) => base.VisitNone(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitNone(None expr, Unit context) => VisitNone(expr);
    /// <summary>
    /// Visit <see cref="Op"/>.
    /// </summary>
    internal protected virtual TExprResult VisitOp(Op expr) => base.VisitOp(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitOp(Op expr, Unit context) => VisitOp(expr);
    /// <summary>
    /// Visit <see cref="PrimFunctionWrapper"/>.
    /// </summary>
    internal protected virtual TExprResult VisitPrimFunctionWrapper(PrimFunctionWrapper expr) => base.VisitPrimFunctionWrapper(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitPrimFunctionWrapper(PrimFunctionWrapper expr, Unit context) => VisitPrimFunctionWrapper(expr);
    /// <summary>
    /// Visit <see cref="TensorConst"/>.
    /// </summary>
    internal protected virtual TExprResult VisitTensorConst(TensorConst expr) => base.VisitTensorConst(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitTensorConst(TensorConst expr, Unit context) => VisitTensorConst(expr);
    /// <summary>
    /// Visit <see cref="IR.Tuple"/>.
    /// </summary>
    internal protected virtual TExprResult VisitTuple(IR.Tuple expr) => base.VisitTuple(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitTuple(IR.Tuple expr, Unit context) => VisitTuple(expr);
    /// <summary>
    /// Visit <see cref="TupleConst"/>.
    /// </summary>
    internal protected virtual TExprResult VisitTupleConst(TupleConst expr) => base.VisitTupleConst(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitTupleConst(TupleConst expr, Unit context) => VisitTupleConst(expr);
    /// <summary>
    /// Visit <see cref="Var"/>.
    /// </summary>
    internal protected virtual TExprResult VisitVar(Var expr) => base.VisitVar(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitVar(Var expr, Unit context) => VisitVar(expr);
    /// <summary>
    /// Visit <see cref="TIR.Block"/>.
    /// </summary>
    internal protected virtual TExprResult VisitBlock(TIR.Block expr) => base.VisitBlock(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitBlock(TIR.Block expr, Unit context) => VisitBlock(expr);
    /// <summary>
    /// Visit <see cref="TIR.Buffer"/>.
    /// </summary>
    internal protected virtual TExprResult VisitBuffer(TIR.Buffer expr) => base.VisitBuffer(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitBuffer(TIR.Buffer expr, Unit context) => VisitBuffer(expr);
    /// <summary>
    /// Visit <see cref="TIR.LogicalBuffer"/>.
    /// </summary>
    internal protected virtual TExprResult VisitLogicalBuffer(TIR.LogicalBuffer expr) => base.VisitLogicalBuffer(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitLogicalBuffer(TIR.LogicalBuffer expr, Unit context) => VisitLogicalBuffer(expr);
    /// <summary>
    /// Visit <see cref="TIR.PhysicalBuffer"/>.
    /// </summary>
    internal protected virtual TExprResult VisitPhysicalBuffer(TIR.PhysicalBuffer expr) => base.VisitPhysicalBuffer(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitPhysicalBuffer(TIR.PhysicalBuffer expr, Unit context) => VisitPhysicalBuffer(expr);
    /// <summary>
    /// Visit <see cref="TIR.BufferLoad"/>.
    /// </summary>
    internal protected virtual TExprResult VisitBufferLoad(TIR.BufferLoad expr) => base.VisitBufferLoad(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitBufferLoad(TIR.BufferLoad expr, Unit context) => VisitBufferLoad(expr);
    /// <summary>
    /// Visit <see cref="TIR.BufferRegion"/>.
    /// </summary>
    internal protected virtual TExprResult VisitBufferRegion(TIR.BufferRegion expr) => base.VisitBufferRegion(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitBufferRegion(TIR.BufferRegion expr, Unit context) => VisitBufferRegion(expr);
    /// <summary>
    /// Visit <see cref="TIR.BufferStore"/>.
    /// </summary>
    internal protected virtual TExprResult VisitBufferStore(TIR.BufferStore expr) => base.VisitBufferStore(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitBufferStore(TIR.BufferStore expr, Unit context) => VisitBufferStore(expr);
    /// <summary>
    /// Visit <see cref="TIR.For"/>.
    /// </summary>
    internal protected virtual TExprResult VisitFor(TIR.For expr) => base.VisitFor(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitFor(TIR.For expr, Unit context) => VisitFor(expr);
    /// <summary>
    /// Visit <see cref="TIR.IfThenElse"/>.
    /// </summary>
    internal protected virtual TExprResult VisitIfThenElse(TIR.IfThenElse expr) => base.VisitIfThenElse(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitIfThenElse(TIR.IfThenElse expr, Unit context) => VisitIfThenElse(expr);
    /// <summary>
    /// Visit <see cref="TIR.Let"/>.
    /// </summary>
    internal protected virtual TExprResult VisitLet(TIR.Let expr) => base.VisitLet(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitLet(TIR.Let expr, Unit context) => VisitLet(expr);
    /// <summary>
    /// Visit <see cref="TIR.PrimFunction"/>.
    /// </summary>
    internal protected virtual TExprResult VisitPrimFunction(TIR.PrimFunction expr) => base.VisitPrimFunction(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitPrimFunction(TIR.PrimFunction expr, Unit context) => VisitPrimFunction(expr);
    /// <summary>
    /// Visit <see cref="TIR.Sequential"/>.
    /// </summary>
    internal protected virtual TExprResult VisitSequential(TIR.Sequential expr) => base.VisitSequential(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitSequential(TIR.Sequential expr, Unit context) => VisitSequential(expr);
    /// <summary>
    /// Visit <see cref="TIR.Range"/>.
    /// </summary>
    internal protected virtual TExprResult VisitRange(TIR.Range expr) => base.VisitRange(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitRange(TIR.Range expr, Unit context) => VisitRange(expr);
    /// <summary>
    /// Visit <see cref="TIR.IterVar"/>.
    /// </summary>
    internal protected virtual TExprResult VisitIterVar(TIR.IterVar expr) => base.VisitIterVar(expr, default);
    
    /// <inheritdoc/>
    internal protected sealed override TExprResult VisitIterVar(TIR.IterVar expr, Unit context) => VisitIterVar(expr);
}