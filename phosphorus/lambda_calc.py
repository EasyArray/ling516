"""
This module contains the implementation of the lambda calculus.
"""
from ast import AST, Constant, Lambda, NodeTransformer

#pylint: disable=invalid-name

class VariableReplacer(NodeTransformer):
  """
  This class replaces variables in a lambda calculus expression with their values."""
  def __init__(self, context): 
    self.context = context

  def visit_Call(self, node):
    """Handle function calls."""
    func = self.visit(node.func)
    if isinstance(func, Lambda):
      vars = {arg.arg for arg in func.args.args}
      for arg in node.args:
        self.visit(arg)
      context = self.context | dict(zip(vars, node.args))
      #print('new context', context)
      return VariableReplacer(context).visit(func.body)

    self.generic_visit(node)
    return node

  def visit_Lambda(self, node):
    """Handle lambda expressions."""
    # TODO: add capture avoidance
    args = {arg.arg for arg in node.args.args}
    shadowed = {v:self.context.pop(v) for v in args if v in self.context}
    new_node = self.generic_visit(node)
    self.context.update(shadowed)
    return new_node

  def visit_Name(self, node):
    """Handle variable names."""
    if node.id in self.context:
      #print(f'Replacing {node.id} with {self.context[node.id]}, {type(self.context[node.id])}')
      if isinstance(self.context[node.id], AST):
        return self.context[node.id]
      if hasattr(self.context[node.id], 'to_ast'):
        return self.context[node.id].to_ast()
      return Constant(value=str(self.context[node.id]))
    return node