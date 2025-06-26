"""Defines Type, SemVal, and Function classes for the Phosphorus Meaning Engine"""

from ast import parse, unparse, fix_missing_locations, iter_fields, literal_eval, dump
from ast import Lambda, Call, Expression, Tuple, arguments, Name, Constant, IfExp, AST
from functools import reduce
from inspect import getclosurevars
from copy import deepcopy
import builtins
from black import format_str, Mode
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

from IPython import get_ipython

from .logs import logger
from .lambda_calc import VariableReplacer, Simplifier, free_vars, evast

# pylint: disable=logging-fstring-interpolation


# This is a bit of somewhat evil magic python code.
class TypeMeta(type):
  """Metaclass to handle Type.<type> expressions"""
  def __getattr__(cls,s):
    #logger.warning(f'Converting {s} of {cls}')
    left_reduce = reduce(
        lambda l,x:  [cls((l[1],l[0]))] + l[2:] if x=='_' else  [cls(x)] + l,
        s, []
    )
    right_reduce = reduce(lambda a,b: cls((b,a)), left_reduce)
    return cls(right_reduce)

class Type(tuple,metaclass=TypeMeta):
  """Represents a type in the Heim & Kratzer system"""
  def isfunction(self):
    """Returns True if the type is a function type"""
    return len(self) == 2
  
  def input(self):
    """Returns the input type of the function type"""
    return Type(self[0])
  
  def output(self):
    """Returns the output type of the function type"""
    return Type(self[1])
  
  # Allows `x in Type.<type>` to check if x is of type <type>
  def __contains__(self, x):
    return getattr(x,'type', None) == self

  def __repr__(self):
    if len(self) == 1:
      return repr(self[0])  # remove parens from simple types
    return super().__repr__()
  
  def __deepcopy__(self, memo):
    obj = Type(deepcopy(tuple(self)))
    memo[id(self)] = obj
    return obj
               

def takes(f,x):
  try:
    if f.eval() is None or x.eval() is None:
      return False
  except: pass
  f_type = getattr(f, 'type', [])
  x_type = getattr(x, 'type', None)
  if len(f_type) == 2:
    if f_type[0] == x_type:
      try:
        test = PV(f(x) is not None) #TODO problem here inlining f
        return test.eval()
      except Exception as e:
        logger.debug(f'Error in takes: {e} {f}({x})')
        return True
  return False

class PV():
  def __new__(cls, node, closure=lambda:{}, type=None):
    if isinstance(node, PV):      
      return node #Question: Simplify or not?
    if isinstance(node, str):
      node = parse(node, mode='eval').body

    global_vars = {x: globals()[x] for x in free_vars(node) if x in globals()}
    try:
      user_ns = get_ipython().user_ns
    except AttributeError:
      user_ns = globals()
    global_vars |= {x: user_ns[x] for x in free_vars(node) if x in user_ns}
    node = Simplifier(global_vars | getclosurevars(closure).nonlocals).visit(node)

    if isinstance(node, PV):
      return node

    node_class = builtins.type(node)
    dynamic_class = builtins.type(
        node_class.__name__,  # name matches AST node class
        (cls, node_class),
        {}
    )
    try:
      instance = node_class.__new__(dynamic_class)
    except TypeError:
      instance = object.__new__(dynamic_class)


    instance._input_node = node
    return instance

  def __init__(self, node, closure=None, type=None):
    if hasattr(self,'fields_filled'):
      if type is not None:
        self.type = type
      repr(self)
      return

    node = self._input_node
    self.type = getattr(node, 'type', None) if type is None else type
    for field, value in iter_fields(node):
      setattr(self, field, value)
    fix_missing_locations(self)
    self.fields_filled = True
    self.closure = closure
    repr(self)

  def copy(self):
    return PV(unparse(self), closure=self.closure, type=self.type)
  
  def __deepcopy__(self, memo):
    obj = self.copy()
    memo[id(self)] = obj
    return obj

  
  def eval(self):
    return evast(self)

  def __hash__(self):
    if isinstance(self, Constant):
      return hash(self.value)
    return hash(dump(self))
  
  def __eq__(self, other):
    if isinstance(other, AST):
      return dump(self) == dump(other)
    return repr(self) == repr(other)
  
  def __bool__(self):
    return bool(self.eval())
  
  def __len__(self):
    from . import charset
    if isinstance(self, Lambda):
      return len(charset(self))
    raise ValueError(f'Cannot get length of {self}')

  def __repr__(self):
    return unparse(self)
    if not hasattr(self, 'repr'):
      self.repr = unparse(self)
    return self.repr

  def _repr_html_(self):
      code_str = format_str(repr(self), mode=Mode())
      highlighted = highlight(code_str, PythonLexer(), HtmlFormatter(noclasses=True))

      return f"""
      <div style="display: flex; align-items: flex-start;">
          <div style="flex-grow: 1; color: #000;">{highlighted}</div>
          <div style="
              font-family: monospace;
              font-weight: bold;
              background-color: #e5e5ff;
              color: #000;
              padding: 0.2em 0.4em;
              margin-left: 1em;
              margin-top: 0.2em;
              margin-right: 1em;
              border-radius: 4px;
              white-space: nowrap;">
              '{self.type}'
          </div>
      </div>
      """
      # return f"""
      # <div style="position: relative; display: inline-block;">
      #     <div style="position: absolute; top: 0; right: 0.5em;
      #                 transform: translateY(1em);
      #                 font-family: monospace; font-weight: bold;
      #                 background-color: #e5e5ff; color: #000;">
      #         '{self.type}'
      #     </div>
      #     <div style="padding-right: 4em;">{highlighted}</div>
      # </div>
      # """
  def x_repr_html_(self):
      s = format_str(repr(self), mode=Mode())
      s = highlight(s, PythonLexer(), HtmlFormatter(noclasses=True, nowrap=True))
      return f"""{s}
          <span style='float:right; font-family:monospace; margin-right:75px;
                font-weight:bold; background-color:#e5e5ff; color:#000'>
            {self.type}</span>"""

class SemVal:
  """Represents a typed semantic value"""

  def __init__(self, s, stype, string=False):
    self.value = s
    self.type = stype
    self.string = string

  def __eq__(self, value):
    return self.value == value
  
  def __hash__(self):
    return hash(self.value)

  @classmethod
  def create(cls, s, stype):
    """Creates a SemVal from a string and a Type"""
    node = parse(s, mode='eval').body
    if isinstance(node, Call):
      node = VariableReplacer({}).visit(node)
      s = unparse(node)
    if stype.isfunction():
      try:    return Function(s, stype)
      except: pass

    return SemVal(s,stype, isinstance(node, Name))

  def _repr_html_(self):
    return f"""{self}
        <span style='float:right; font-family:monospace; margin-right:75px;
              font-weight:bold; background-color:#e5e5ff; color:#000'>
          {self.type}</span>"""

  def __repr__(self):
    return repr(self.value) if self.string else str(self.value)

  def domain(self):
    """Returns the domain of a function, to be overridden by Function"""
    return set() # There's no domain for nonfunctions

  def to_ast(self):
    """Returns the AST of the value, for use an IPython AST transformer"""
    return parse(repr(self), mode='eval').body

class Function(SemVal):
  """Represents a function SemVal"""

  def __init__(self, s, stype, context=None):
    if not stype.isfunction():
      raise ValueError(f'Invalid type for "{s}": {stype}')

    node = parse(s, mode='eval').body
    match node:
      case Lambda(args=arguments(args=args),
                  body=(Tuple(elts=(guard, value))
                        | IfExp(test=guard, body=value, orelse=Constant(value=None)))):
        self.vars = tuple(arg.arg for arg in args)
        guard_expr = unparse(guard)
        value_expr = unparse(value)
        self.restriction = guard_expr
        value = f"{value_expr} if {guard_expr} else None"
      case Lambda(args=arguments(args=args), body=body):
        self.vars = tuple(arg.arg for arg in args)
        self.restriction = None
        value = unparse(body)
      case _:
        raise ValueError(f'Invalid lambda expression: {s}')

    self.context = context if context else {}
    super().__init__(value, stype)

  def __call__(self, *args):
    #logger.warning(f'Calling {repr(self)} with {args}')
    # NOTE TO SELF: some logic is replicated in VariableReplacer
    node = parse(self.value, mode='eval').body
    context = dict(zip(self.vars, args))
    node = VariableReplacer(context).visit(node)
    #logger.debug('Context %s, Node %s', context, dump(node))

    out_type = self.type.output()
    if isinstance(node, Lambda):
      try:
        return Function(unparse(node), out_type)
      except ValueError as e:
        logger.error(f'Error in function [λ{self.vars} . {self.value}] (type {self.type}): {e}')
        raise e

    try:
      exprnode = Expression(body=node)
      fix_missing_locations(exprnode)
      code = compile(exprnode, '<string>', 'eval')
      value = eval(code, get_ipython().user_ns, self.context | context) # pylint: disable=eval-used
      return value
    except Exception as e:
      logger.debug(f'Error evaluating {unparse(node)}: {e}')
      value = unparse(node)
      if out_type.isfunction():
        raise ValueError(f'Output of function [λ{self.vars} . {self.value}] (type {self.type})\n'
                         f'\tis not type {out_type}: {value}')

    return SemVal.create(value, out_type)

  def domain(self):
    return self.type.input()

  def __str__(self):
    vars = ','.join(self.vars)
    return f'λ{vars} . {self()}'

  def __repr__(self):
    from json import dumps
    vars = ','.join(self.vars)
    value = dumps(f'lambda {vars}: {self.value}')
    out = f'Function({value}, Type({self.type}))'
    return out
  
  def to_ast(self):
    #TODO: fix functions to work with repr() asts
    # essentially, we need SemVal's to recursively infect higher structures so
    # SemVal(...) and SemVal(...) returns the correct SemVal instead of just the second
    return parse(f'lambda {",".join(self.vars)}: {self.value}', mode='eval').body
