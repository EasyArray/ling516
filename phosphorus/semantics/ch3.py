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

# ── hack so that “import phosphorus…” works when run as a script ──
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]   # go up: semantics → phosphorus → repo root
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
# ─────────────────────────────────────────────────────────────────


from phosphorus.semantics.interpret import Interpreter, UNDEF
from phosphorus.syntax.tree           import Tree
from phosphorus.core.phivalue         import PhiValue
from phosphorus.core.stypes           import Type, takes

__all__ = ["register_ch3"]
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ——————————————————————————————————————————————
# Rule registration
# ——————————————————————————————————————————————

def register_ch3(interp: Interpreter) -> None:
  """Attach TN, NN, FA rules to *interp* with TN > NN > FA priority."""

  # ——— TN ————————————————————————————————————————————
  @interp.rule(index=0)
  def TN(alpha: str = ""):
    """Terminal Node: lexical lookup of *alpha* (string token)."""
    return interp.lookup(alpha)

  # ——— NN ————————————————————————————————————————————
  @interp.rule(index=1)
  def NN(child: PhiValue):
    """Non‑branching Node: pass child meaning unchanged."""
    return child

  # ——— FA ————————————————————————————————————————————
  @interp.rule(index=2)
  def FA(beta: PhiValue, gamma: PhiValue):
    """Functional Application (order determined by `takes`)."""
    if takes(beta, gamma):
      fn, arg = beta, gamma
    elif takes(gamma, beta):
      fn, arg = gamma, beta
    else:
      return UNDEF

    # Build AST `fn(arg)` with *names* so PhiValue captures env
    call_ast = ast.Call(
      func=ast.Name(id="fn", ctx=ast.Load()),
      args=[ast.Name(id="arg", ctx=ast.Load())],
      keywords=[],
    )
    result_type = fn.stype.range  # safe given `takes` check
    return PhiValue(call_ast, stype=result_type)

# ——————————————————————————————————————————————
# Self‑contained sanity tests
# ——————————————————————————————————————————————

def _build_lexicon():
  """Toy lexicon with constant, intransitive and transitive verbs."""
  # john / mary  — individuals
  john = PhiValue(ast.parse("JOHN", mode="eval").body, stype=Type.e)
  mary = PhiValue(ast.parse("MARY", mode="eval").body, stype=Type.e)

  # runs — λx.RUN(x)
  runs = PhiValue(
    ast.parse("lambda x=e: RUN(x).t", mode="eval").body
  )

  # loves — λy.λx.LOVE(x,y)
  loves = PhiValue(
    ast.parse("lambda y=e: (lambda x=e: LOVE(x,y).t)", mode="eval").body,
  )

  return {
    "john": john,
    "mary": mary,
    "runs": runs,
    "loves": loves,
  }


def _self_test():
  import logging

  # Configure the root logger to print DEBUG (and higher) to stdout
  logging.basicConfig(level=logging.DEBUG, format="%(name)s:%(levelname)s: %(message)s")

  lex = _build_lexicon()
  interp = Interpreter(lexicon=lex)
  register_ch3(interp)

  # 1. TN  ------------------------------------------------------
  t1 = Tree.fromstring("(N John)")
  m1 = interp.interpret(t1)
  print(t1.pformat(margin=50))
  print("⇒", m1)
  assert isinstance(m1, PhiValue) and m1.expr.id == "JOHN"

  # 2. NN  ------------------------------------------------------
  t2 = Tree.fromstring("(DP (N John))")
  m2 = interp.interpret(t2)
  print(t2.pformat(margin=50))
  print("⇒", m2)
  assert m2 is m1

  # 3. FA (intrans)  -------------------------------------------
  t3 = Tree.fromstring("(S (N John) (V runs))")
  m3 = interp.interpret(t3)
  print(t3.pformat(margin=50))
  print("⇒", m3)
  assert isinstance(m3, PhiValue)
  assert isinstance(m3.expr, ast.Call)

  # 4. FA (trans)  ---------------------------------------------
  t4 = Tree.fromstring("(S (N John) (VP (V loves) (N Mary)))")
  m4 = interp.interpret(t4)
  print(t4.pformat(margin=50))
  print("⇒", m4)
  assert isinstance(m4, PhiValue)

  print("ch3: all tests passed ✓")

if __name__ == "__main__":
  _self_test()
