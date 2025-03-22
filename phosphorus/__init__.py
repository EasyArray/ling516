"""
The Phosphorus Meaning Engine defines classes and functions to interpret
natural language semantics in the style of Heim & Kratzer (1998)."""

# pylint: disable=invalid-name

from ast import NodeTransformer, parse, dump, unparse, fix_missing_locations, keyword
from ast import Lambda, Constant, Call, Name, Attribute, Load, FunctionDef, arguments, Expression
from inspect import getclosurevars
from string import ascii_uppercase

from IPython import get_ipython
from .logs import logger, console_handler, memory_handler, logging
from .semval import SemVal, Type, Function, PV, takes
from .meaning import Meaning, VACUOUS

class ExprTransformer(NodeTransformer):
  """Transforms expressions of the form '...'.<type> into SemVal objects."""

  def visit_Call(self, node):
    """Handle PV(...) calls."""
    self.generic_visit(node)
    match node:
      case Call(func=Name(id='PV') as func, args=[code, *args], keywords=kwargs): 
        type=None
      case Call(func=Attribute(value=Name(id='PV') as func, attr=type, ctx=Load()), 
                args=[code, *args], keywords=kwargs):
        pass
      case _: return node

    node.func = func
    code_string = unparse(code).strip()
    if not isinstance(code, Lambda):
      code = Lambda(args=arguments([], [], None, [], [], None, []), body=code)
    node.args = [Constant(value=code_string), code, *args]
    if type is not None:
      kwargs = [k for k in kwargs if k.arg != 'type']
      kwargs.append(keyword(arg='type', 
                            value=Attribute(value=Name(id='Type', ctx=Load()), attr=type, ctx=Load())))
    node.keywords = kwargs
    fix_missing_locations(node)
    #print('PV call:', unparse(node), dump(node))

    return node

  def visit_Attribute(self, node):
    """Handle attributes of the form '...'.<type>."""
    self.generic_visit(node)
    match node:
      case Attribute(value=Constant() as code_string, attr=type) if not hasattr(str, type):
        try:
          code = parse(code_string.value, mode='eval').body
        except SyntaxError as e:
          print('Error parsing PV code:', code_string.value, e)
          return node
      case Attribute(value=Lambda() as code, attr=type):
        code_string = Constant(value=unparse(code).strip())
      case _: return node

    if not isinstance(code, Lambda):
      code = Lambda(args=arguments([], [], None, [], [], None, []), body=code)
    node.value = Name(id='Type', ctx=Load())
    out = Call(func=Name(id='PV', ctx=Load()), 
               args=[code_string, code, node], keywords=[])
    fix_missing_locations(out)
    #print('Attribute', unparse(out), dump(out))
    return out

# Add the ExprTransformer to the IPython AST transformers
ip_asts = get_ipython().ast_transformers
while(ip_asts and type(ip_asts[-1]).__name__ == "ExprTransformer"):
  del ip_asts[-1]
ip_asts.append(ExprTransformer())


class Predicate(set):
  """A set of tuples representing a predicate."""
  def __call__(self, *args):
    if any(a is None for a in args):
      return None
    if any(a not in DOMAIN for a in args):
      raise TypeError(f'Predicates only take individuals in the DOMAIN, got: {args}')
    return int(args in self) # converts True/False to 1/0
  
  def __repr__(self):
    return '<Predicate: %s>' % super().__repr__()


DOMAIN = [PV(repr(c), type=Type.e) for c in ascii_uppercase]

def charset(f, domain = None):
  if domain is None:
    domain = DOMAIN
  return {c for c in domain if f(c)}

def iota(f, domain = None):
  return tuple(charset(f,domain))[0]

def single(s):
  return len(s)==1

def empty(s):
  return len(s)==0


# Splash screen
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
