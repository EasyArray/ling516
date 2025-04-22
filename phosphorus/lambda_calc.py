"""
This module contains the implementation of the lambda calculus.
"""
import builtins
from ast import *
from string import ascii_lowercase
from inspect import signature
from itertools import zip_longest
from copy import deepcopy
from IPython import get_ipython
from .logs import logger

#pylint: disable=invalid-name
  

AST_ENV = {}
exec('from ast import *', AST_ENV)

def toast(x, type = None, code_string=True):
  if isinstance(x, AST):
    #logger.debug('toasting AST %s %s', dump(x), builtins.type(x))
    out = x
  else:
    if not code_string:
      x = repr(x)
    #print('toasting', x)
    out = parse(x, mode='eval').body

  if type is not None:
    out.type = type

  return out

def evast(node, context={}, env=None):
  expr_node = Expression(body=node)
  fix_missing_locations(expr_node)
  compiled  = compile(expr_node, '<AST>', 'eval')
  env       = AST_ENV | (get_ipython().user_ns if env is None else env)
  #logger.debug('EVAST IN %s', unparse(node))
  evaled    = eval(compiled, env, context.copy())
  #logger.debug('EVAST OUT %s (%s)', evaled, type(evaled))
  return evaled

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
  def __init__(self, context=None, env=None, **kwargs):
    # TODO: introduce globals here? pare down based on free_vars in vist?
    self.context = {} if context is None else context
    self.context.update(kwargs)
    self.env = env
    #logger.debug('SIMPLIFIER %s', {k:(unparse(v) if isinstance(v,AST) else v) for k, v in self.context.items()})

  def visit(self, node):
    node = toast(node)
    if not isinstance(node, expr):
      return node

    logger.debug('visiting %s %s %s', unparse(node), dump(node), get_type(node))
    #logger.debug('CONTEXT %s', {k:(unparse(v) if isinstance(v,AST) else v) for k, v in self.context.items()})
    try:
      evaled    = evast(node, self.context, self.env)
      node.evaled = True
      if getattr(node, 'inlined', False) and evaled is not None and not isinstance(evaled, AST):
        raise TypeError(f'Unable to inline code {unparse(node)}')
      toasted   = toast(evaled, code_string=False)
      toasted.evaled = True
      logger.debug('Evaluated %s to %s (%s)', unparse(node), unparse(toasted), type(evaled))
      logger.debug('DUMP %s', dump(node))
      #logger.debug('CONTEXT %s', {k:(unparse(v) if isinstance(v,AST) else v) for k, v in self.context.items()})
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
    #logger.warning('AFTER generic visit %s\n%s', unparse(node), dump(node))
    try:
      _, out_type = get_type(node.func)
    except (ValueError, TypeError):
      out_type = None

    match node.func:
      case Lambda(args=arguments(args=params), body=body):
        params  = [arg.arg for arg in params]
        args    = [arg for arg in node.args]
        context = self.context | dict(zip(params, args))
        body = deepcopy(body)
        return toast(Simplifier(context).visit(body), out_type)

      case _ if not hasattr(node, 'inlined'):
        # TRY to call the function on the ast nodes of its arguments instead of the arguments
        # themselves. TODO: check args individually? or update context instead?
        try:
          func = evast(node.func, self.context, self.env)
          sig = signature(func)
          ast_typed = [issubclass(p.annotation, AST) for p in sig.parameters.values()]
          #logger.debug('AST TYPED %s %s', ast_typed, sig.parameters)
          if any(ast_typed):
            arg_ast_pairs = zip_longest(node.args, ast_typed, fillvalue=ast_typed[-1])
            ast_params = [toast(dump(arg)) if is_ast and not getattr(arg, 'evaled', False) else arg 
                          for arg,is_ast in arg_ast_pairs]
            logger.debug('AST PARAMS %s %s', 
                         [(arg, getattr(arg, 'evaled', False)) for arg in node.args], 
                         [unparse(p) for p in ast_params])
            new_node = toast(Call(func=node.func, args=ast_params, keywords=node.keywords), get_type(node))
            fix_missing_locations(new_node)
            new_node.inlined = True
            out = toast(self.visit(new_node), out_type)
            return out
        except (SyntaxError, Exception) as e:
          logger.debug('Error evaluating Call: %s', e)
    return toast(node, get_type(node) if get_type(node) else out_type)

  def visit_Name(self, node):
    #print('NAME visiting Name', node.id, node.ctx)
    if node.id in self.context:
      out = self.context[node.id] #Need to visit here?
      try:
        return toast(out, get_type(node))
      except (SyntaxError, Exception) as e:
        #logger.debug('Error toasting %s: %s', node.id, e)
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

    #logger.warning('LAMBDA: %s', unparse(node))
    #logger.warning('FREE IN BODY %s', free_in_body)
    #logger.warning('FREE IN REPLACEMENTS %s', free_in_replacements)
    alpha_conversions = {}
    for arg in node.args.args:
      if arg.arg in free_in_replacements:
        new_param = new_var(arg.arg, free_in_body | free_in_replacements)
        alpha_conversions[arg.arg] = Name(id=new_param, ctx=Load())
        #logger.warning('REPLACING %s with %s, ctx %s', arg.arg, new_param, alpha_conversions[arg.arg])
        arg.arg = new_param
    #logger.warning('CONVERSIONS %s', {k:(unparse(v) if isinstance(v,AST) else v) for k, v in alpha_conversions.items()})
    if alpha_conversions:
      #logger.warning('OLD LAMBDA %s', unparse(node))
      Simplifier(alpha_conversions, {}).generic_visit(node)
      #logger.warning('NEW LAMBDA %s', unparse(node))
    return Simplifier(context).generic_visit(node)

  def visit_BinOp(self, node):
    self.generic_visit(node)
    match node:
      case BinOp(op=BitOr(), left=Dict() as left, right=Dict() as right):
        combined = (
            dict(zip(map(lambda x: getattr(x,'value',x), left.keys), left.values))
            | dict(zip(map(lambda x: getattr(x,'value',x), right.keys), right.values))
        )
        return toast(Dict(keys=list(map(lambda x: toast(x,code_string=False), combined.keys())),
                          values=list(combined.values())), get_type(node))
    return node

  def visit_BoolOp(self, node):
    self.generic_visit(node)
    match node:
      case BoolOp(op=And(), values=values):
        if any(isinstance(v, Constant) and not v.value for v in values):
          return toast(Constant(value=False), get_type(node))
        if all(isinstance(v, Constant) and v.value for v in values):
          return toast(Constant(value=True), get_type(node))
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