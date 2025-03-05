"""
The Phosphorus Meaning Engine defines classes and functions to interpret
natural language semantics in the style of Heim & Kratzer (1998)."""

from ast import NodeTransformer, parse, dump, unparse, fix_missing_locations
from ast import Lambda, Constant, Call, Name, Attribute, Load, FunctionDef, arguments, Expression

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
      case _: return node
    out = Call(func=Attribute(value=Name(id='SemVal', ctx=Load()),
                              attr='create', ctx=Load()),
                args=[semval, node], keywords=[])
    #print(dump(out))
    return out

# Add the ExprTransformer to the IPython AST transformers
ip_asts = get_ipython().ast_transformers
while(ip_asts and type(ip_asts[-1]).__name__ == "ExprTransformer"):
  del ip_asts[-1]
ip_asts.append(ExprTransformer())


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
