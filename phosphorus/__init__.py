"""
The Phosphorus Meaning Engine defines classes and functions to interpret
natural language semantics in the style of Heim & Kratzer (1998)."""

# pylint: disable=invalid-name

from string import ascii_uppercase
from IPython import get_ipython

from .semantics.interpret import Interpreter, defined
from .syntax.tree import Tree
from .core.phivalue import PhiValue
from .core.stypes import Type, takes
from .core.constants import UNDEF, VACUOUS

# Install the backtick DSL for PhiValue literals
from .dsl import backtick

# Install the CSS for rendering PhiValues in Jupyter
#from .core.display import inject_css
#inject_css()

DOMAIN = [PhiValue(repr(c), stype=Type.e) for c in ascii_uppercase]

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


def charset(f, domain = None):
  if domain is None:
    domain = DOMAIN
  return {c for c in domain if PV('f(c)', lambda:f(c))}

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

        Welcome to the Phosphorus Meaning Engine v4
        Created by Ezra Keshet (EzraKeshet.com)

""")
