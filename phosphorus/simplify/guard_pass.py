"""
phosphorus/simplify/guard_pass.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GuardFolder – a *Simplify* pass that normalises guard syntax written
with the ``%`` operator.

Rules
-----
* ``φ % False``   → ``UNDEF``
* ``φ % True``    → ``φ``
* If the *right‑hand side* can be **evaluated in the current env** to a
  boolean constant, fold as above (Option 1: cheap eval).
* ``(φ % ψ) is not UNDEF``  →  ``ψ``
* ``defined(φ % ψ)``        →  ``ψ``

This pass should run *after* macro‑expansion and beta‑reduction, and
before any static checks of undefinedness.
"""

from __future__ import annotations

import ast

from .passes import SimplifyPass   # base class provides .env (ChainMap)

# sentinel name for undefined
UNDEF_NAME = "UNDEF"

# ---------------------------------------------------------------------------
# GuardFolder pass
# ---------------------------------------------------------------------------

class GuardFolder(SimplifyPass):
  """Fold `%`‑guard AST patterns into canonical forms (Option 1)."""

  TESTS = [
    ("phi % False",          UNDEF_NAME),
    ("phi % True",           "phi"),
    ("defined(phi % psi)",   "psi"),
    ("(phi % psi) is not UNDEF", "psi"),
  ]

  TEST_ENV = {
    'UNDEF': ast.Name(id=UNDEF_NAME, ctx=ast.Load()),
    'defined': lambda x: x is not UNDEF,   # dummy; folded away
  }

  # ---------- helper: try static eval of RHS -----------------------
  def _eval_bool(self, expr: ast.AST):
    try:
      code = compile(ast.Expression(expr), filename="<ast>", mode="eval")
      val = eval(code, self.env)
      if isinstance(val, bool):
        return val
    except Exception:
      pass
    return None

  # ---------- φ % ψ  ------------------------------------------------
  def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
    self.generic_visit(node)

    match node:
      # φ % False  -> UNDEF
      case ast.BinOp(op=ast.Mod(), right=ast.Constant(value=False)):
        return ast.Name(id=UNDEF_NAME, ctx=ast.Load())

      # φ % True   -> φ
      case ast.BinOp(left=lhs, op=ast.Mod(), right=ast.Constant(value=True)):
        return lhs

      # φ % ψ  where ψ evals to boolean constant
      case ast.BinOp(left=lhs, op=ast.Mod(), right=rhs):
        val = self._eval_bool(rhs)
        if val is False:
          return ast.Name(id=UNDEF_NAME, ctx=ast.Load())
        if val is True:
          return lhs

    return node

  # ---------- (φ % ψ) is not UNDEF  →  ψ ---------------------------
  def visit_Compare(self, node: ast.Compare) -> ast.AST:
    self.generic_visit(node)
    match node:
      case ast.Compare(
        left=ast.BinOp(left=_, op=ast.Mod(), right=rhs),
        ops=[ast.IsNot()],
        comparators=[ast.Name(id=UNDEF_NAME, ctx=ast.Load())]
      ):
        return rhs
    return node

  # ---------- defined(φ % ψ)  →  ψ ---------------------------------
  def visit_Call(self, node: ast.Call) -> ast.AST:
    self.generic_visit(node)

    match node:
      case ast.Call(
        func=ast.Name(id="defined", ctx=ast.Load()),
        args=[ast.BinOp(op=ast.Mod(), right=rhs)],
        keywords=[],
      ):
        return rhs
    return node
