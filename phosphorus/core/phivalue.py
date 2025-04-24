"""phosphorus.core.phivalue
---------------------------------
Light‑weight wrapper around a Python AST that stores an optional
semantic type and beta‑reduces the AST at construction time.

All heavy work (pretty HTML, richer type inference) lives elsewhere.
"""

from __future__ import annotations

import ast
from collections import ChainMap
from typing import Any, Optional

# ---------------------------------------------------------------------------
#  Internal one‑shot imports (static, no dynamic import_module)
# ---------------------------------------------------------------------------

from phosphorus.simplify import simplify           # noqa: E402  (local import style)
from phosphorus.simplify.utils import capture_env  # noqa: E402

try:
  from phosphorus.core.infer import infer_type  # type: ignore
except ImportError:  # typing not ready yet
  def infer_type(node: ast.AST, env: dict):  # type: ignore
    return None  # placeholder

# ---------------------------------------------------------------------------
#  PhiValue
# ---------------------------------------------------------------------------

class PhiValue:
  """AST + optional semantic type + captured environment."""

  __slots__ = ("expr", "stype", "_env")

  # ---------------------------------------------------------------------
  #  construction
  # ---------------------------------------------------------------------

  def __init__(self, expr: ast.AST, *, stype: Optional[Any] = None):
    # 1. capture *caller* environment (skip=1 frame up)
    env = capture_env(skip=1)  # returns a ChainMap already

    # 2. beta‑reduce / macro‑expand via functional simplify
    #    (simplify currently expects a *string*; switch when API changes)
    simplified_ast = simplify(expr, env=env) if isinstance(expr, ast.AST) else ast.parse(simplify(ast.unparse(expr), env=env), mode="eval").body  # type: ignore[arg-type]

    # 3. store
    self.expr: ast.AST = simplified_ast
    self._env: ChainMap[str, Any] = env

    # 4. assign semantic type (may stay None)
    self.stype = stype if stype is not None else infer_type(self.expr, env)

  # ---------------------------------------------------------------------
  #  functional behaviour
  # ---------------------------------------------------------------------

  def __call__(self, *args: "PhiValue") -> "PhiValue":
    call_ast = ast.Call(func=self.expr,
                        args=[a.expr for a in args],
                        keywords=[])
    return PhiValue(call_ast)  # stype inferred inside constructor

  # ---------------------------------------------------------------------
  #  evaluation helpers
  # ---------------------------------------------------------------------

  def eval(self) -> Any:
    """Evaluate the stored expression in its captured environment."""
    # Python's eval requires a real dict for globals
    env_dict = dict(self._env)
    # ensure builtins are available
    env_dict.setdefault('__builtins__', __builtins__)
    code = compile(ast.Expression(self.expr), filename="<phivalue>", mode="eval")
    return eval(code, env_dict)

  # ---------------------------------------------------------------------
  #  dunder utilities
  # ---------------------------------------------------------------------

  def __bool__(self):
    return bool(self.eval())

  def __hash__(self):
    return hash((ast.dump(self.expr, annotate_fields=False), self.stype))

  def __eq__(self, other):
    if isinstance(other, PhiValue):
      return (ast.dump(self.expr, annotate_fields=False) ==
              ast.dump(other.expr, annotate_fields=False)) and self.stype == other.stype
    return NotImplemented

  def __repr__(self):
    return ast.unparse(self.expr)

  # Jupyter rich repr stub (real HTML lives in display.py)
  def _repr_html_(self):
    return f"<code>{ast.unparse(self.expr)}</code>  <small>{self.stype}</small>"

# ---------------------------------------------------------------------------
#  rudimentary tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  id_ast = ast.parse("lambda x: x", mode="eval").body
  id_pv = PhiValue(id_ast)
  two_pv = PhiValue(ast.parse("2", mode="eval").body)

  assert id_pv(two_pv).eval() == 2
  assert id_pv == id_pv
  assert isinstance(hash(id_pv), int)
  print("✅ PhiValue sanity tests passed.")
