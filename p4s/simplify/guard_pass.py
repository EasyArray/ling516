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
* ``defined(...)``    →  ``True`` otherwise
* ``(fn % G)(arg)`` →  ``(fn)(arg) % G``

This pass should run *after* macro‑expansion and beta‑reduction, and
before any static checks of undefinedness.
"""

from __future__ import annotations

import ast
from collections import deque

from p4s.core.constants import UNDEF   # sentinel for undefined values
from .passes import SimplifyPass   # base class provides .env (ChainMap)

# sentinel name for undefined
UNDEF_NAME = str(UNDEF)

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
    ("defined(lambda x: x)", "True"),  # New test for defined(lambda...)
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
      
      # defined(foo) → True if foo is not a BinOp %
      # in the future may want to try to eval first
      case ast.Call(
        func=ast.Name(id="defined", ctx=ast.Load()),
#        args=[ast.Lambda()],
#        keywords=[],
      ):
        return ast.Constant(value=True)
      
      case ast.Call(
        func=ast.BinOp(left=lhs, op=ast.Mod(), right=rhs)
      ):
        node.func = lhs
        return ast.BinOp(left=node, op=ast.Mod(), right=rhs)
        
    return node


# ---------------------------------------------------------------------------
# Remove Duplicate Guard Pass
# ---------------------------------------------------------------------------

def guard_key(node: ast.AST) -> str:
  """
  Return a hashable, location-independent representation of *node*.

  We rely on ast.dump(), suppressing lineno/col_offset etc.,
  which is deterministic from 3.8 onward.
  """
  return ast.dump(node, annotate_fields=True, include_attributes=False)

class RemoveDuplicateGuards(SimplifyPass):
  """
  Delete inner occurrences of “… % guard” when *guard* is
  structurally identical to one already in force.

    (A % G) % G        → A % G
    (A % G1 % G2) % G1 → A % G1 % G2
  """

  # we keep a stack of *keys* for the guards that dominate the
  # current traversal position
  def __init__(self, env=None):
    super().__init__(env)
    self._active: deque[str] = deque()

  # ---------- helper ------------------------------------------------
  def _is_redundant(self, guard: ast.AST) -> bool:
    return guard_key(guard) in self._active

  # ---------- core visitor ------------------------------------------
  def visit_BinOp(self, node: ast.BinOp):
    # only interested in `%`
    match node:
      case ast.BinOp(op=ast.Mod()):
        # (1) First simplify the *guard* expression on the right
        node.right = self.visit(node.right)

        # (2) Check redundancy
        key = guard_key(node.right)
        if key in self._active:
          # drop the whole “… % guard”
          return self.visit(node.left)

        # (3) Keep it and recurse into the left with guard active
        self._active.append(key)
        node.left = self.visit(node.left)
        self._active.pop()
        return node

      case _:
        return self.generic_visit(node)