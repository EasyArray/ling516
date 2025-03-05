# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring,invalid-name

from IPython import get_ipython

# This class converts expressions of the form '...'.type into
# `SemVal`s (defined below)
from ast import *
class ExprTransformer(NodeTransformer):
  def visit_Attribute(self, node):
    self.generic_visit(node)
    match node:
      case Attribute(value=Constant() as semval,
                    attr=stype) if not hasattr(str, stype):
        node.value = Name(id='Type', ctx=Load())
      case _: return node
    out = Call(func=Attribute(value=Name(id='SemVal', ctx=Load()),
                              attr='create', ctx=Load()),
                args=[semval, node], keywords=[])
    #print(dump(out))
    return out

ip_asts = get_ipython().ast_transformers
while(ip_asts and type(ip_asts[-1]).__name__ == "ExprTransformer"):
  del ip_asts[-1]
ip_asts.append(ExprTransformer())

# This class replaces variables in python code with
# values provided by a context.
class VariableReplacer(NodeTransformer):
  def __init__(self, context): 
    self.context = context

  def visit_Call(self, node):
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
    # TODO: add capture avoidance
    args = {arg.arg for arg in node.args.args}
    shadowed = {v:self.context.pop(v) for v in args if v in self.context}
    new_node = self.generic_visit(node)
    self.context.update(shadowed)
    return new_node

  def visit_Name(self, node):
    if node.id in self.context:
      #print(f'Replacing {node.id} with {self.context[node.id]}, {type(self.context[node.id])}')
      if isinstance(self.context[node.id], AST):
        return self.context[node.id]
      if hasattr(self.context[node.id], 'to_ast'):
        return self.context[node.id].to_ast()
      return Constant(value=str(self.context[node.id]))
    return node
  
# This is a bit of somewhat evil magic python code. It handles the
# Type.<type> expressions
class TypeMeta(type):
  def __getattr__(cls,s):
    from functools import reduce
    left_reduce = reduce(
        lambda l,x:  [cls((l[1],l[0]))] + l[2:] if x=='_' else  [cls(x)] + l,
        s, []
    )
    right_reduce = reduce(lambda a,b: cls((b,a)), left_reduce)
    return cls(right_reduce)

# Here is the main Type class, a subclass of tuple
class Type(tuple,metaclass=TypeMeta):
  def isfunction(self): return len(self) == 2
  def input(self):      return Type(self[0])
  def output(self):     return Type(self[1])
  def __contains__(self, x):  return x.type == self

  def __repr__(self):
    if len(self) == 1:  return repr(self[0])
    return super().__repr__()

class SemVal():
  def __init__(self, s, stype):
    self.value = s
    self.type = stype

  @classmethod
  def create(cls, s, stype):
    node = parse(s, mode='eval').body
    if isinstance(node, Call):
      node = VariableReplacer({}).visit(node)
      s = unparse(node)
    if stype.isfunction():
      try:    return Function(s, stype)
      except: pass
    return SemVal(s,stype)

  def _repr_html_(self):
    return f"""{self}
        <span style='float:right; font-family:monospace; margin-right:75px;
              font-weight:bold; background-color:#e5e5ff; color:#000'>
          {self.type}</span>"""

  def __repr__(self): return str(self.value)
  def domain(self): return set() # There's no domain for nonfunctions
  def to_ast(self): return parse(repr(self), mode='eval').body

class Function(SemVal):
  def __init__(self, s, stype):
    if not stype.isfunction():
      raise ValueError(f'Invalid type for "{s}": {stype}')
    self.type = stype
    node = parse(s, mode='eval').body
    match node:
      case Lambda(args=arguments(args=args), body=body):
        self.vars = tuple(arg.arg for arg in args)
        self.value = unparse(body)
      case _:
        msg = f'Invalid lambda expression: {s}'
        raise ValueError(msg)

  def __call__(self, *args):
    # NOTE TO SELF: some logic is replicated in VariableReplacer
    node = parse(self.value, mode='eval').body
    context = dict(zip(self.vars, args))
    node = VariableReplacer(context).visit(node)
    #print(dump(node))

    out_type = self.type.output()
    if isinstance(node, Lambda):
      try:
        return Function(unparse(node), out_type)
      except ValueError as e:
        logger.error(f'Error in function [λ{self.vars} . {self.value}] (type {self.type}): {e}')
        raise e

    try:
      exprnode = Expression(body=node)
      fix_missing_locations(expr)
      code = compile(exprnode, '<string>', 'eval')
      value = eval(code, get_ipython().user_ns)
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
    return parse(f'lambda {",".join(self.vars)}: {self.value}', mode='eval').body

import logging
# ANSI escape codes for colors
COLORS = {
    "DEBUG": "\033[90m",  # Gray
    "INFO": "",            # Default color
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",   # Red
    "CRITICAL": "\033[41m\033[97m",  # White on Red Background
    "RESET": "\033[0m"    # Reset color
}

class ColorFormatter(logging.Formatter):
  """Custom formatter to add color based on log level."""
  def format(self, record):
    log_color = COLORS.get(record.levelname, COLORS["RESET"])
    message = super().format(record)
    return f"{log_color}{message}{COLORS['RESET']}"  # Wrap message in color codes

from logging.handlers import MemoryHandler

# Create a stream handler (prints logs to console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(ColorFormatter("%(message)s"))

# Create a memory handler to buffer logs below WARNING level
memory_handler = MemoryHandler(capacity=1000, target=console_handler, flushLevel=logging.CRITICAL)
memory_handler.setLevel(logging.INFO)

# Set up root logger
logger = logging.getLogger("BufferedLogger")
logger.setLevel(logging.DEBUG)  # Capture all logs internally
logger.addHandler(console_handler)
logger.propagate = False # Prevent double logging in Colab


class Meaning(dict):
  indent = ''
  indent_chars = '   '
  def print(self, *args, level=logging.INFO):
    msg = self.indent + ' '.join(map(str, args))
    logger.log(level, msg)

  # Defines the behavior of m[]
  def __getitem__(self, k): return self.interpret(k)

  # Just look up a word in the lexicon
  def lookup(self, word):   return super().get(word, None)

  # Main interpretation function
  def interpret(self, alpha):
    try:
      m = self
      if not m.indent: 
        m.print() # Skip a line before the first output
        self.prev_logger_level = console_handler.level
        memory_handler.buffer.clear()
      m.print('Interpreting', alpha)
      m.indent += m.indent_chars

      if isinstance(alpha, tuple):
        alpha = list(alpha)
      if isinstance(alpha, list):
        if len(alpha) == 0:
          raise ValueError(f'Node {alpha} has no children')
        vacuous = [x for x in alpha if m.quiet(m[x]) is None]
        if vacuous:
          m.print('Removing vacuous items:', vacuous, level=logging.WARNING)
          alpha[:] = (x for x in alpha if x not in vacuous)
      
      if not alpha:
        m.print('No non-vacuous children in node', alpha, level=logging.WARNING)
        value, rule = None, 'NN'
      else:
        value, rule = self.rules(alpha)
        if value is rule is None:
          children = ' and '.join(map(str, alpha))
          raise ValueError(f'No rule found to combine {children}')

      m.indent = m.indent[:-len(m.indent_chars)]
      m.print('=>', alpha, '=', value, f'\t({rule})')
      return value
    except Exception as e:
      self.indent = ''
      console_handler.setLevel(self.prev_logger_level)
      memory_handler.setLevel(logging.CRITICAL)
      m.print(f'!!! Error interpreting node {alpha}:\n {e}', level=logging.ERROR)
      if len(memory_handler.buffer) > 0:
        m.print('Previously silenced output:', level=logging.ERROR)
        memory_handler.flush()
      raise e

  def rules(m, alpha): # pylint: disable=no-self-argument
    value, rule = None, None
    match alpha:      # Note: m.quiet(  ) turns off printing
      # PM
      case (beta, gamma) if m.quiet(  m[gamma].type == m[beta].type == Type.et ):
        rule = 'PM'
        pm_f = Function('lambda f : lambda g: lambda x: f(x) and g(x)', Type.et_et_et)
        value = pm_f(m[beta])(m[gamma])

      # FA
      case (beta, gamma) if m.quiet(  m[gamma] in m[beta].domain()  ):
        rule = 'FA'
        value = m[beta](m[gamma])
      case (gamma, beta) if m.quiet(  m[gamma] in m[beta].domain()  ):
        rule = 'AF'
        value = m[beta](m[gamma])

      # NN
      case (beta,):
        rule = 'NN'
        value = m[beta]

      # TN
      case str():
        rule = 'TN'
        value = m.lookup(alpha)

    return value, rule

  # This (somewhat evil) code handles the m.quiet( ) functionality
  def __getattr__(self, s):
    if 'quiet'.startswith(s):
      prev = console_handler.level
      console_handler.setLevel(logging.WARNING)
      if memory_handler not in logger.handlers:
        logger.addHandler(memory_handler)
      def run(condition):
        console_handler.setLevel(prev)
        if memory_handler in logger.handlers:
          logger.removeHandler(memory_handler)
        return condition
    else:
      def run(condition):
        return condition
    return run

print(r"""
             _    _                  _    _
            | |  | |                | |  | |
           _| |_ | |__   ___  ___  _| |_ | |__   ___  _ __ _   _  ____
          /     \| '_ \ / _ \/ __|/     \| '_ \ / _ \| '__| | | |/ ___)
         ( (| |) ) | | | (_) \__ ( (| |) ) | | | (_) | |  | |_| ( (__
          \_   _/|_| |_|\___/|___/\_   _/|_| |_|\___/|_|   \__,_|\__ \
            | |                     | |                            _) )
            |_|                     |_|                           (__/

        Welcome to the Phosphorus Meaning Engine v3
        Created by Ezra Keshet (EzraKeshet.com)

""")
