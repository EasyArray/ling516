"""
The Phosphorus Meaning Engine defines classes and functions to interpret
natural language semantics in the style of Heim & Kratzer (1998)."""

# pylint: disable=invalid-name

from string import ascii_uppercase
import ast
from IPython import get_ipython

from .semantics.interpret import Interpreter, defined, rule
from .syntax.tree import Tree
from .core.phivalue import PhiValue
from .core.stypes import *
from .core.constants import UNDEF, VACUOUS

# Install the backtick DSL for PhiValue literals
from .dsl import backtick
from .dsl.backtick import *

# Install the CSS for rendering PhiValues in Jupyter
#from .core.display import inject_css
#inject_css()

DOMAIN = [PhiValue(repr(c), stype=Type.e) for c in ascii_uppercase]

class Predicate(set):
  """A set of tuples representing a predicate."""
  @staticmethod
  def _canon_individual(item):
    # Treat symbolic individuals like B as equivalent to string individuals 'B'.
    if isinstance(item, PhiValue) and isinstance(item.expr, ast.Name):
      return PhiValue(repr(item.expr.id), stype=item.stype)
    return item

  @classmethod
  def _canon_tuple(cls, item):
    if not isinstance(item, tuple):
      item = (item,)
    return tuple(cls._canon_individual(x) for x in item)

  def __contains__(self, item):
    # Use == (which honours PhiValue.__eq__) instead of hash-based set lookup,
    # so that e.g. the string 'B' and PhiValue('B') are treated as the same individual.
    # Also normalise bare individuals to 1-tuples so `'B' in BLACK` works like `('B',) in BLACK`.
    item = self._canon_tuple(item)
    return any(item == self._canon_tuple(tup) for tup in set.__iter__(self))

  def __call__(self, *args):
    args = self._canon_tuple(args)
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
  return {c for c in domain if f(c)}

def singular(f, domain = None):
  if domain is None:
    domain = DOMAIN
  stype = getattr(f, 'stype', None)
  if stype is not None:
    match stype:
      case (domain_t, Type.t) if domain_t == Type.e or getattr(domain_t, 'is_unknown', False):
        pass
      case _:
        return False

  def safe_apply(x):
    try:
      return f(x)
    except (NameError, TypeError, AttributeError):
      return UNDEF

  def to_truth(y):
    if isinstance(y, PhiValue):
      try:
        y = y.eval()
      except (NameError, TypeError, AttributeError, ValueError):
        return 0
    if y in (None, UNDEF):
      return 0
    try:
      return int(bool(y))
    except (TypeError, ValueError):
      return 0

  return sum(to_truth(safe_apply(x)) for x in domain) == 1

def empty(f, domain = None):
  if domain is None:
    domain = DOMAIN
  return not any(f(x) for x in domain)

def iota(f, domain = None):
  return tuple(charset(f,domain))[0]

def single(s):
  return len(s)==1

#def empty(s):
#  return len(s)==0


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
