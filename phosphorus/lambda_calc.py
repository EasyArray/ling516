"""
This module contains the implementation of the lambda calculus.
"""
from ast import *
from string import ascii_lowercase
from IPython import get_ipython
from .logs import logger

#pylint: disable=invalid-name

AST_ENV = {}
exec('from ast import *', AST_ENV)

def toast(x, type = None, code_string=True):
  if isinstance(x, AST):
    out = x
  else:
    if not code_string:
      x = repr(x)
    #print('toasting', x)
    out = parse(x, mode='eval').body

  if type is not None:
    out.type = type

  return out

def get_type(x):
  if hasattr(x, 'type'):
    return x.type

#TODO: to really be complete, need to track ctx Store() vars too.
def free_vars(node):
  match node:
    case Name(id=name, ctx=Load()):
      return {name}
    case Lambda(args=arguments(args=params), body=body):
      return free_vars(body) - {arg.arg for arg in params}
    case _:
      return {v for child in iter_child_nodes(node) for v in free_vars(child)}

def new_var(old, avoid='', vars=ascii_lowercase):
  try:
    before, after = vars.split(old)
    return next(c for c in after+before if c not in avoid)
  except (ValueError, StopIteration):
    old = '_' + old
    return new_var(old, avoid, vars) if old in avoid else old

class Simplifier(NodeTransformer):
  def __init__(self, context=None, **kwargs):
    # TODO: introduce globals here? pare down based on free_vars in vist?
    self.context = {} if context is None else context
    self.context.update(kwargs)
    #print('SIMPLIFIER', {k:(unparse(v) if isinstance(v,AST) else v) for k, v in self.context.items()})

  def visit(self, node):
    node = toast(node)
    if not isinstance(node, expr):
      return node

    print('visiting', unparse(node), dump(node), get_type(node))
    print('CONTEXT', self.context.keys())#{k:(unparse(v) if isinstance(v,AST) else v) for k, v in self.context.items()})
    try:      
      expr_node = Expression(body=node)
      fix_missing_locations(expr_node)
      compiled  = compile(expr_node, '<AST>', 'eval')
      env       = AST_ENV | get_ipython().user_ns
      evaled    = eval(compiled, env, self.context.copy())
      if getattr(node, 'inlined', False) and not isinstance(evaled, AST):
        raise TypeError(f'Unable to inline code {unparse(node)}')
      toasted   = toast(evaled, code_string=False)
      logger.debug('Evaluated %s to %s', unparse(node), unparse(toasted))
      return toasted
    except (SyntaxError,Exception) as e:
      logger.debug('Error evaluating %s %s', unparse(node), dump(node))
      logger.debug('ERROR: %s', e)
      #old_node = unparse(node)
      node = super().visit(node)
      # Recursive call here?
      return node


  def visit_Call(self, node):
    logger.debug('CALL visiting Call %s\n%s', unparse(node), dump(node))
    self.generic_visit(node)

    try:
      _, out_type = get_type(node.func)
    except (ValueError, TypeError):
      out_type = None

    match node.func:
      case Lambda(args=arguments(args=params), body=body):
        params  = [arg.arg for arg in params]
        args    = [arg for arg in node.args]
        context = self.context | dict(zip(params, args))
        return toast(Simplifier(context).visit(body), out_type)

      case _ if not hasattr(node, 'inlined'):
        # TRY to call the function on the ast nodes of its arguments instead of the arguments
        # themselves. TODO: check args individually? or update context instead?
        try:
          ast_params = [arg if isinstance(arg, Constant) else toast(dump(arg)) for arg in node.args]
          logger.debug('AST PARAMS %s %s', [arg for arg in node.args], [unparse(p) for p in ast_params])
          new_node = toast(Call(func=node.func, args=ast_params, keywords=node.keywords), get_type(node))
          fix_missing_locations(new_node)
          new_node.inlined = True
          out = toast(self.visit(new_node), out_type)
          return out
        except (SyntaxError, Exception) as e:
          print('Error evaluating Call', e)
    return toast(node, get_type(node) if get_type(node) else out_type)

  def visit_Name(self, node):
    #print('NAME visiting Name', node.id, node.ctx)
    if node.id in self.context:
      out = self.context[node.id]
      try:
        return toast(out, get_type(node))
      except (SyntaxError, Exception) as e:
        logger.debug('Error toasting %s: %s', node.id, e)
        return node
    return node

  def visit_Lambda(self, node):
    params  = [arg.arg for arg in node.args.args]
    context = {k:v for k,v in self.context.items() if k not in params}
    free_in_body = free_vars(node.body)
    free_in_replacements = {
        f for value in context.values() if isinstance(value, AST)
          for f in free_vars(value)
    }

    for arg in node.args.args:
      if arg.arg in free_in_replacements:
        new_param = new_var(arg.arg, free_in_body | free_in_replacements)
        context[arg.arg] = Name(id=new_param, ctx=Load())
        arg.arg = new_param
    return Simplifier(context).generic_visit(node)

  def visit_BinOp(self, node):
    self.generic_visit(node)
    match node:
      case BinOp(op=BitOr(), left=Dict() as left, right=Dict() as right):
        combined = (
            dict(zip(left.keys, left.values)) | dict(zip(right.keys, right.values))
        )
        return toast(Dict(keys=list(combined.keys()),
                          values=list(combined.values())), get_type(node))
    return node

  def visit_Subscript(self, node):
    self.generic_visit(node)
    #print('SUBSCRIPT', unparse(node), dump(node))
    match node:
      case Subscript(value=Dict() as d, slice=Constant() as c):
        mapping = {k.value: v for k,v in zip(d.keys, d.values) if isinstance(k, Constant)}
        value = mapping.get(c.value, node)
        return toast(value, get_type(node))
    return node

class VariableReplacer(NodeTransformer):
  """
  This class replaces variables in a lambda calculus expression with their values."""
  def __init__(self, context): 
    self.context = context

  def visit_Call(self, node):
    """Handle function calls."""
    func = self.visit(node.func)
    if isinstance(func, Lambda):
      params = {arg.arg for arg in func.args.args}
      for arg in node.args:
        self.visit(arg)
      context = self.context | dict(zip(params, node.args))
      #logger.debug('new context: %s', context)
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
      #logger.warning(f'Replacing {node.id} with {self.context[node.id]}, {type(self.context[node.id])}')
      if isinstance(self.context[node.id], AST):
        return self.context[node.id]
      if hasattr(self.context[node.id], 'to_ast'):
        return self.context[node.id].to_ast()
      #return Constant(value=str(self.context[node.id]))
      return parse(repr(self.context[node.id]), mode='eval').body
    return node