"""
This module defines the Meaning class, which is used to interpret the meaning of a natural 
language expression.
"""

from .logs import logger, console_handler, memory_handler, logging
from .semval import Function, Type

class Meaning(dict):
  """The Meaning class interprets the meaning of a natural language expression."""
  indent = ''
  indent_chars = '   '
  def print(self, *args, level=logging.INFO):
    """Logs a message with correct indentation."""
    msg = self.indent + ' '.join(map(str, args))
    logger.log(level, msg)

  # This allows us to use m[] for interpretation
  def __getitem__(self, k):
    return self.interpret(k)

  # Just look up a word in the lexicon
  def lookup(self, word):
    """Used to simply look up a word in the lexicon without further interpretation"""
    return super().get(word, None)

  def interpret(self, alpha):
    """Interprets the meaning of a natural language expression alpha."""
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
    """Defines standard rules for combining the meanings of the
    children of a node alpha. Meant to be overridden if different rules are wanted."""

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
