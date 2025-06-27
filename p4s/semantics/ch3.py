"""
phosphorus.semantics.ch3
~~~~~~~~~~~~~~~~~~~~~~~~
Heim & Kratzer (1998) **Chapter 3** rules
========================================
* **TN** – Terminal Node (lexical lookup)
* **NN** – Non‑Branching Node (parent = child)
* **FA** – Functional Application

The rules are registered via `register_ch3(interp)` so that students can
mix‑and‑match rule sets.

```
from phosphorus.semantics.interpreter import Interpreter
from phosphorus.semantics.ch3         import register_ch3

interp = Interpreter(lexicon=my_lexicon)
register_ch3(interp)
```
"""

import ast
import logging

from p4s.semantics.interpret import Interpreter, UNDEF
from p4s.syntax.tree           import Tree
from p4s.core.phivalue         import PhiValue
from p4s.core.stypes           import takes

__all__ = ["register_ch3"]

# ——————————————————————————————————————————————
# Rule registration
# ——————————————————————————————————————————————

def register_ch3(interp: Interpreter) -> None:
  """Attach TN, NN, FA rules to *interp* with TN > NN > FA priority."""

  # ——— TN ————————————————————————————————————————————
  @interp.rule()
  def TN(alpha: str = ""):
    """Terminal Node: lexical lookup of *alpha* (string token)."""
    return interp.lookup(alpha)

  # ——— NN ————————————————————————————————————————————
  @interp.rule()
  def NN(child: PhiValue):
    """Non‑branching Node: pass child meaning unchanged."""
    return child

  # ——— FA ————————————————————————————————————————————
  @interp.rule()
  def FA(beta: PhiValue, gamma: PhiValue):
    """Functional Application (order determined by `takes`)."""
    if takes(beta, gamma):
      fn, arg = beta, gamma
    elif takes(gamma, beta):
      fn, arg = gamma, beta
    else:
      return UNDEF

    result_type = fn.stype.range  # safe given `takes` check
    return PhiValue('fn(arg)', stype=result_type)

# ——————————————————————————————————————————————
# Self‑contained sanity tests
# ——————————————————————————————————————————————

def _build_lexicon():
  """Toy lexicon with constant, intransitive and transitive verbs."""
  return {
    "john":   PhiValue('JOHN.e'),
    "mary":   PhiValue('MARY.e'),
    "runs":   PhiValue("lambda x=e: RUN(x).t"),
    "loves":  PhiValue("lambda y=e: (lambda x=e: LOVE(x,y).t)"),
  }


def _self_test():
  interp = Interpreter(lexicon=_build_lexicon())
  register_ch3(interp)

  samples = [
    "(N John)",
    "(DP (N John))",
    "(S (N John) (V runs))",
    "(S (N John) (VP (V loves) (N Mary)))",
  ]

  from IPython.display import display
  for src in samples:
    tree = Tree.fromstring(src)
    meaning = interp.interpret(tree)
    print(tree)
    print(f'=>{meaning}[{meaning.stype}]')
    print()

if __name__ == "__main__":
  _self_test()
