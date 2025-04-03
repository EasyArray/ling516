"""
This module defines the Meaning class, which is used to interpret the meaning of a natural 
language expression.
"""

from ast import AST, unparse, dump, parse
from nltk import Tree, ImmutableTree
from .logs import logger, console_handler, memory_handler, logging
from .semval import Function, Type, PV

class _VacuousSentinel:
    def __repr__(self):
        return "VACUOUS"
VACUOUS = _VacuousSentinel()

class ImmutableDict(tuple):
  """A subclass of tuple to indicate the original object was a dictionary."""

def make_hashable(obj):
  """Converts an object to a hashable form."""
  if isinstance(obj, dict):
    return ImmutableDict((key, make_hashable(value)) for key, value in obj.items())
  if isinstance(obj, Tree):
    return ImmutableTree.convert(obj)
  if isinstance(obj, (list,tuple)):
    return tuple(make_hashable(x) for x in obj)
  if isinstance(obj, AST):
    return dump(obj)
  return obj

def make_mutable(obj):
  """Converts an object to a mutable form."""
  if isinstance(obj, ImmutableTree):
    return Tree.convert(obj)
  if isinstance(obj, ImmutableDict):
    return dict((key, make_mutable(value)) for key, value in obj)
  if isinstance(obj, (tuple,list)):
    return [make_mutable(x) for x in obj]
  return obj


class Meaning(dict):
  """The Meaning class interprets the meaning of a natural language expression."""

  memo = {}
  indent = ''
  indent_chars = '   '
  def print(self, *args, level=logging.INFO):
    """Logs a message with correct indentation."""
    msg = self.indent + ' '.join(map(str, args))
    logger.log(level, msg)

  def __call__(self, *args):
    class ParamMeaning(type(self)):
      def __getitem__(_, k):
        return self[k:args]
    return ParamMeaning(self)
  
  def __repr__(self):
    return object.__repr__(self)
  
  # This allows us to use m[] for interpretation
  def __getitem__(self, k):
    args = ()
    if isinstance(k, slice):
      args = k.stop
      k = k.start
    return self.i(k, args)
  
  def i(self, k, *args:AST):
    if not self.indent:
        self.memo.clear()
        logger.debug('Cleared Memo buffer: %s', self.memo)
    k = make_hashable(k)
    hargs = make_hashable(args)
    if (k, hargs) not in self.memo:
      try:
        self.memo[k,hargs] = self.interpret(k, *args)
        logger.debug('Memoizing value for (%s, %s): %s', k, hargs, self.memo[k,hargs])
      except Exception as e:
        self.print('Error interpreting (%s, %s): %s' % (k, args, e))
        return None
    else:
      logger.debug('Using memoized value for (%s, %s): %s', k, hargs, self.memo[k,hargs])
    out = self.memo[k,hargs]
    if not self.indent:
      try:
        evaled = out.eval()
        parse(repr(evaled))
        out = evaled
      except: pass
    return out
    
  # Just look up a word in the lexicon
  def lookup(self, word):
    """Used to simply look up a word in the lexicon without further interpretation"""
    out = super().get(word, None)
    if isinstance(out, PV):
      out = out.copy()
    logger.debug('Lookup for %s: %s (%s)', word, out, type(out))
    return out

  def interpret(self, alpha, *args):
    """Interprets the meaning of a natural language expression alpha."""
    try:
      m = self
      shortalpha = getattr(alpha, 'label', lambda:alpha)()
      m.print('Interpreting', shortalpha, 'with parameters:',
              [(unparse(arg), getattr(arg, 'type', None)) 
               if isinstance(arg, AST) else arg for arg in args])
      m.indent += m.indent_chars

      if isinstance(alpha, (tuple, list)): #TODO: do this once at the beginning?
        if len(alpha) == 0:
          raise ValueError(f'Node {alpha} has no children')
        # alpha = make_mutable(alpha)
        # vacuous = [x for x in alpha if m.i(x,*args) is None]
        # if vacuous:
        #   m.print('Removing vacuous items:', vacuous, level=logging.WARNING)
        #   #logger.warning('With vacuous items removed: %s', [x for x in alpha if x not in vacuous])
        #   alpha[:] = (x for x in alpha if x not in vacuous)
        #   #logger.warning('New alpha: %s, Vacuous: %s, v[0] in vac:%s', alpha, vacuous, vacuous[0] in vacuous)
        # alpha = make_hashable(alpha)

      
      if isinstance(alpha, (tuple, list)) and len(alpha) == 0:
        m.print('No non-vacuous children in node', alpha, level=logging.WARNING)
        value, rule = None, 'NN'
      else:
        value, rule = self.rules(alpha, *args)
        if value is None and rule not in ('TN', 'NN'): #fix
          children = ' and '.join(map(str, alpha))
          #raise ValueError(f'No rule found to combine {children}')
          m.print(f'No rule found to combine {children}', level=logging.ERROR)

      m.indent = m.indent[:-len(m.indent_chars)]
      m.print('=>', shortalpha, '=', value, f" type: {getattr(value, 'type', None)}\t({rule})")
      return value
    except Exception as e:
      #self.indent = ''
      m.print(f'!!! Error interpreting node {alpha}:', level=logging.ERROR)
      m.print(e, level=logging.ERROR)
      #import traceback
      #traceback.print_exc()
      self.indent = self.indent.removesuffix(self.indent_chars)
      raise e

  def rules(m, alpha, *args): # pylint: disable=no-self-argument
    """Defines standard rules for combining the meanings of the
    children of a node alpha. Meant to be overridden if different rules are wanted."""

    value, rule = None, None
    match alpha:      
      # PM
      case (beta, gamma) if m[gamma:args].type == m[beta:args].type == Type.et:
        rule = 'PM'
        pm_f = Function('lambda f : lambda g: lambda x: f(x) and g(x)', Type.et_et_et)
        value = pm_f(m[beta:args])(m[gamma:args])

      # FA
      case (beta, gamma) if  m[gamma:args] in m[beta:args].domain() :
        rule = 'FA'
        value = m[beta:args](m[gamma:args])
      case (gamma, beta) if  m[gamma:args] in m[beta:args].domain() :
        rule = 'AF'
        value = m[beta:args](m[gamma:args])

      # NN
      case (beta,):
        rule = 'NN'
        value = m[beta:args]

      case 't'|'he'|'she'|'it':
        rule = 'TP'
        value = args[0] if len(args) == 1 else args
      
      # TN
      case str():
        rule = 'TN'
        value = m.lookup(alpha)

    return value, rule
  
  def quiet(self, x):
    """For backwards compatibility"""
    return x
