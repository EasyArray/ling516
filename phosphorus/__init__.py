"""
The Phosphorus Meaning Engine defines classes and functions to interpret
natural language semantics in the style of Heim & Kratzer (1998)."""

from ast import NodeTransformer, parse, dump, unparse, fix_missing_locations
from ast import Lambda, Constant, Call, Name, Attribute, Load, FunctionDef, arguments, Expression
from inspect import getclosurevars

from IPython import get_ipython
from .logs import logger, console_handler, memory_handler, logging
from .semval import SemVal, Type, Function
from .meaning import Meaning

class ExprTransformer(NodeTransformer):
  """Transforms expressions of the form '...'.<type> into SemVal objects."""

  def visit_Attribute(self, node):
    """Handle attributes of the form '...'.<type>."""
    self.generic_visit(node)
    match node:
      case Attribute(value=Constant() as semval,
                    attr=stype) if not hasattr(str, stype):
        node.value = Name(id='Type', ctx=Load())
        return Call(func=Attribute(value=Name(id='SemVal', ctx=Load()),
                                   attr='create', ctx=Load()),
                    args=[semval, node], keywords=[])

      case Attribute(value=Lambda() as lam, attr=stype):
        semval = Constant(value=unparse(lam).strip())
        node.value = Name(id='Type', ctx=Load())
        context_node = Attribute(value=Call(func=Name(id='getclosurevars', ctx=Load()),
                                            args=[lam], keywords=[]),
                                  attr='nonlocals', ctx=Load())

        out = Call(func=Name(id='Function', ctx=Load()),
                    args=[semval, node, context_node], keywords=[])
        fix_missing_locations(out)
        #print(dump(out))
        return out

    return node

# Add the ExprTransformer to the IPython AST transformers
ip_asts = get_ipython().ast_transformers
while(ip_asts and type(ip_asts[-1]).__name__ == "ExprTransformer"):
  del ip_asts[-1]
ip_asts.append(ExprTransformer())


class Predicate(set):
  """A set of tuples representing a predicate."""
  def __call__(self, *args):
    return int(args in self) # converts True/False to 1/0


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
