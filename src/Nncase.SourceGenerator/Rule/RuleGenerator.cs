﻿using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;
namespace Nncase.SourceGenerator.Rule;

[Generator]
internal class RuleGenerator : ISourceGenerator
{
    public void Execute(GeneratorExecutionContext context)
    {
        if (context.SyntaxContextReceiver is not RuleReceiver receiver)
            return;
        receiver.Diagnostics.ForEach(d => context.ReportDiagnostic(d));
        var grouped_classes = (from cand in receiver.Candidates
                               select cand.classSymobl.ContainingNamespace)
                               .Distinct()
                               .ToDictionary(s => s, s => new List<ClassDeclarationSyntax>());

        foreach (var cand in receiver.Candidates)
        {
            // 1. consturct statements
            var statements = new List<StatementSyntax>();
            foreach (var parameterSymbol in cand.methodSymbol.Parameters)
            {
                if (parameterSymbol.Equals(receiver.IMatchResultSymobl))
                    continue;
                if (parameterSymbol.Name == "result")
                {
                    context.ReportDiagnostic(Diagnostic.Create(RecriverUtil.MethodParamError, Location.None,
                     cand.classSymobl.ToDisplayString(),
                     parameterSymbol.Name,
                     $"Parameter Name Can Not Be result."));
                    return;
                }
                statements.Add(
                  ParseStatement($"var {parameterSymbol.Name} = ({parameterSymbol.Type.ToDisplayString()})result[\"{parameterSymbol.Name}\"];")
                );
            }
            statements.Add(
              ParseStatement($"return {cand.methodSymbol.Name}({string.Join(",", cand.methodSymbol.Parameters.Where(p => !p.Type.Equals(receiver.IMatchResultSymobl)).Select(p => p.Name))});")
            );

            // 2. consturct wrapper method.
            var method = MethodDeclaration(ParseTypeName("Nncase.IR.Expr?"), Identifier("GetReplace"))
                        .WithParameterList(ParseParameterList("(IMatchResult result)"))
                        .WithModifiers(TokenList(Token(SyntaxKind.PublicKeyword), Token(SyntaxKind.OverrideKeyword)))
                        .WithBody(Block(statements));

            // 3. add classes 
            grouped_classes[cand.classSymobl.ContainingNamespace].Add(
              cand.classDeclaration
              .WithIdentifier(Identifier(cand.classSymobl.Name))
              .WithMembers(SingletonList<MemberDeclarationSyntax>(method))
              .WithAttributeLists(new SyntaxList<AttributeListSyntax>() { })
              );
        }

        if (grouped_classes.Count == 0)
            return;

        var namespaces = (from kv in grouped_classes
                          select NamespaceDeclaration(ParseName(kv.Key.ToDisplayString()))
                                .AddMembers(kv.Value.ToArray()));
        var compilationUnit = CompilationUnit().
                AddMembers(namespaces.ToArray()).
                AddUsings(
                  UsingDirective(ParseName("Nncase")),
                  UsingDirective(ParseName("Nncase.IR")),
                  UsingDirective(ParseName("Nncase.PatternMatch"))
                ).
                NormalizeWhitespace();
        context.AddSource("Generated.Rules", SyntaxTree(compilationUnit, encoding: Encoding.UTF8).GetText());
    }

    public void Initialize(GeneratorInitializationContext context) => context.RegisterForSyntaxNotifications(() => new RuleReceiver());

}