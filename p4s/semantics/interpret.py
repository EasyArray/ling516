"""
phosphorus.semantics.interpreter
--------------------------------
Bottom‑up compositional interpreter à la Heim & Kratzer.

Rules
~~~~~
A rule is a callable whose **first positional** parameter is the syntax
node currently being interpreted (a ``Tree`` or a leaf token).  The
remaining positional parameters receive the *non‑vacuous* meanings of the
node’s children.

Return values
-------------
* a denotation (usually a ``PhiValue``) → success, stops rule search
* ``UNDEF`` → rule did not apply; interpreter tries the next rule

Sentinels
~~~~~~~~~
* **VACUOUS** – ignorable material (e.g. punctuation, unknown lexical)
* **UNDEF**   – undefined result *or* rule‑skip marker

Logging (``logging.DEBUG``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* removal of *VACUOUS* children
* a node collapsing to *VACUOUS*
* arity mismatches or exceptions inside rule trials
"""

import logging
from inspect import Parameter, signature
from typing import Any, Callable, Mapping

from p4s.syntax.tree import Tree
from p4s.core.phivalue import PhiValue
from p4s.core.constants import UNDEF, VACUOUS


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["Interpreter", "defined"]

# ——————————————————————————————————————————————
# Helpers
# ——————————————————————————————————————————————

def defined(value: PhiValue) -> bool:
  """Check if a PhiValue is defined (not UNDEF)."""
  if isinstance(value, PhiValue):
    return value.eval() is not UNDEF
  return False

# ——————————————————————————————————————————————
# Interpreter
# ——————————————————————————————————————————————

class Interpreter:
  """Bundle of a *lexicon* and an ordered list of composition *rules*."""

  def __init__(self,
               lexicon: Mapping[str, Any] | None = None,
               *,
               rules: list[Callable] | None = None) -> None:
    self.lexicon: dict[str, Any] = dict(lexicon or {})
    self.rules: list[Callable] = list(rules or [])

  # ――― helpers ――――――――――――――――――――――――――――――――――――
  def lookup(self, word: str):
    """Return lexicon entry for *word* (case‑insensitive) or VACUOUS."""
    return self.lexicon.get(word.lower(), VACUOUS)

  def add_rule(self, fn: Callable, *, index: int | None = None) -> None:
    if index is None:
      self.rules.append(fn)
    else: 
      self.rules.insert(index or 0, fn)

  def rule(self, fn: Callable | None = None, *, index: int | None = None):
    """Decorator: ``@interp.rule`` registers *fn* as a rule."""
    if fn is None:
      return lambda f: self.rule(f, index=index)
    self.add_rule(fn, index=index)
    return fn

  # ――― public API ――――――――――――――――――――――――――――――――
  def interpret(self, tree: Tree, *extra_args):
    val = self._compute(tree)
    if extra_args and callable(val):
      return val(*extra_args)
    return val

  # ――― core recursive worker ――――――――――――――――――――――
  def _compute(self, node):
    """Compute denotation for *node* (``Tree`` **or** leaf token)."""

    child_vals = []
    if isinstance(node, Tree):
      # 1. Gather child denotations (empty for leaf tokens)
      child_vals = [self._compute(ch) for ch in node]
  
      # 2. Filter VACUOUS children (log indices removed)
      non_vac: list[Any] = []
      for idx, val in enumerate(child_vals):
        if val is VACUOUS:
          if isinstance(node, Tree):
            logger.debug("Removed VACUOUS child %d of %s", idx, node.label())
        else:
          non_vac.append(val)

      # 3. Collapse Tree with no semantic children
      if not non_vac:
        logger.debug("Node %s became VACUOUS (no non‑vacuous children)", node.label())
        node.sem = VACUOUS
        return VACUOUS
        
      child_vals = non_vac

    # Try rules in order
    for rule in self.rules:
      val = self._try_rule(rule, node, child_vals)
      if val is not UNDEF:
        try:
          node.sem = val
        except: pass
        return val

    # 5. No rule succeeded
    if isinstance(node, Tree):
      logger.debug("No rule succeeded on node %s", node.label())
      node.sem = UNDEF
    return UNDEF

  # ――― safe rule invocation ――――――――――――――――――――――
  @staticmethod
  def _try_rule(rule: Callable, node, child_args: list[Any]):
    """Attempt to apply *rule* to *child_args* and optional *alpha=node* keyword."""
    # 1. Arity check based on positional parameters only
    try:
      sig = signature(rule)
      params = list(sig.parameters.values())
      fixed_pos = [p for p in params if p.kind in (Parameter.POSITIONAL_ONLY,
                                                   Parameter.POSITIONAL_OR_KEYWORD)]
      has_var_pos = any(p.kind == Parameter.VAR_POSITIONAL for p in params)
      min_needed = sum(1 for p in fixed_pos if p.default is Parameter.empty)
      max_allowed = float('inf') if has_var_pos else len(fixed_pos)
      if not (min_needed <= len(child_args) <= max_allowed):
        lbl = node.label() if isinstance(node, Tree) else str(node)
        logger.debug("Arity mismatch for rule %s on node %s", rule.__name__, lbl)
        return UNDEF
    except (TypeError, ValueError):
      # Could not introspect; assume it might work
      pass

    # 2. Determine whether to pass alpha as keyword
    kwargs: dict[str, Any] = {}
    try:
      sig = signature(rule)
      if 'alpha' in sig.parameters:
        kwargs['alpha'] = node
    except (TypeError, ValueError):
      pass

    # 3. Call the rule with only child_args and optional alpha kwarg
    try:
      return rule(*child_args, **kwargs)
    except Exception as exc:
      logger.debug("Rule %s raised %s", rule.__name__, exc)
      return UNDEF