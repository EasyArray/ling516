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
* ``defined(...)``          →  ``True`` / ``False`` when statically decidable
* ``φ and (ψ % G)``         →  ``(φ and ψ) % G`` (conservative, conjunction only)
* ``(fn % G)(arg)`` →  ``(fn)(arg) % G``
* ``f(..., (a % G), ...)`` → ``f(..., a, ...) % G``

This pass should run *after* macro‑expansion and beta‑reduction, and
before any static checks of undefinedness.
"""

from __future__ import annotations

import ast

from p4s.core.constants import UNDEF   # sentinel for undefined values
from .passes import SimplifyPass   # base class provides .env (ChainMap)

# sentinel name for undefined
UNDEF_NAME = str(UNDEF)


def _free_vars(node: ast.AST, bound: set[str] | None = None) -> set[str]:
  """Collect free variable names in *node* (Load context only)."""
  if bound is None:
    bound = set()

  match node:
    case ast.Name(id=name, ctx=ast.Load()):
      return set() if name in bound else {name}
    case ast.Lambda(args=args, body=body):
      inner_bound = set(bound)
      inner_bound.update(a.arg for a in args.args)
      inner_bound.update(a.arg for a in args.kwonlyargs)
      if args.vararg is not None:
        inner_bound.add(args.vararg.arg)
      if args.kwarg is not None:
        inner_bound.add(args.kwarg.arg)
      return _free_vars(body, inner_bound)
    case _:
      out: set[str] = set()
      for child in ast.iter_child_nodes(node):
        out.update(_free_vars(child, bound))
      return out

# ---------------------------------------------------------------------------
# GuardFolder pass
# ---------------------------------------------------------------------------

class GuardFolder(SimplifyPass):
  """Fold `%`‑guard AST patterns into canonical forms (Option 1)."""

  TESTS = [
    ("phi % False", UNDEF_NAME),
    ("phi % True", "phi"),
    ("defined(phi % psi)", "psi"),
    ("(phi % psi) is not UNDEF", "psi"),
    ("defined(lambda x: x)", "True"),
    ("defined(defined(x))", "True"),
    ("defined(MAN(J))", "True"),
    ("MAN(J) % defined(MAN(J))", "MAN(J)"),
    ("defined(iota(z))", "defined(iota(z))"),
    ("phi and (psi % G)", "(phi and psi) % G"),
    ("(phi % G) and psi", "(phi and psi) % G"),
    ("(phi % G1) and (psi % G2)", "(phi and psi) % G1 % G2"),
    ("G and (phi % G)", "(G and phi) % G"),
    (
      "lambda x: p(x) % g % defined(p(x))",
      "(lambda x: p(x) % defined(p(x))) % g",
    ),
    ("f(x % g)", "f(x) % g"),
    (
      "KILLED(x, iota(z) % singular(z))",
      "KILLED(x, iota(z)) % singular(z)",
    ),
    ("lambda x: p(x) % g", "(lambda x: p(x)) % g"),
    ("iota(lambda x: p(x) % g)", "iota(lambda x: p(x)) % g"),
  ]

  TEST_ENV = {
    'UNDEF': ast.Name(id=UNDEF_NAME, ctx=ast.Load()),
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

  def _is_assumed_defined_name(self, name: str) -> bool | None:
    """Return static definedness for bare names under the lexical assumptions."""
    if name == UNDEF_NAME:
      return False
    if name in self.env:
      return self.env.get(name) is not UNDEF
    if name[:1].isupper():
      return True
    return None

  def _defined_status(self, expr: ast.AST) -> bool | None:
    """Return True, False, or None for statically decidable definedness."""
    match expr:
      case ast.Name(id=name, ctx=ast.Load()):
        return self._is_assumed_defined_name(name)

      case ast.Constant():
        return True

      case ast.Lambda():
        return True

      case ast.Call(
        func=ast.Name(id="defined", ctx=ast.Load()),
        args=[_],
        keywords=[],
      ):
        # defined(...) itself is total even when its argument is not.
        return True

      case ast.Call(
        func=ast.Name(id=fn_name, ctx=ast.Load()),
        args=args,
        keywords=[],
      ):
        if fn_name == UNDEF_NAME:
          return False
        if fn_name[:1].isupper() and all(self._defined_status(arg) is True for arg in args):
          return True
        if fn_name in self.env and self.env.get(fn_name) is UNDEF:
          return False
        return None

      case ast.BinOp(left=lhs, op=ast.Mod(), right=rhs):
        lhs_status = self._defined_status(lhs)
        rhs_bool = self._eval_bool(rhs)
        if lhs_status is False or rhs_bool is False:
          return False
        if lhs_status is True and rhs_bool is True:
          return True
        return None

      case _:
        return None

  @staticmethod
  def _collect_guard_chain(node: ast.AST) -> tuple[ast.AST, list[ast.AST]]:
    """Return (payload, guards) where guards are in application order."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mod):
      return node, []

    payload, guards = GuardFolder._collect_guard_chain(node.left)
    rhs_payload, rhs_guards = GuardFolder._collect_guard_chain(node.right)
    guards.append(rhs_payload)
    guards.extend(rhs_guards)
    return payload, guards

  @staticmethod
  def _rebuild_guard_chain(payload: ast.AST, guards: list[ast.AST]) -> ast.AST:
    out = payload
    for guard in guards:
      out = ast.BinOp(left=out, op=ast.Mod(), right=guard)
    return out

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
  def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
    self.generic_visit(node)

    # Hoist guards from top-level conjunction terms so duplicate guard
    # elimination can see and collapse repeated guards.
    #
    # TODO: revisit with full presupposition projection/cancellation.
    # For example, G and (phi % G) can cancel the guard, but we keep it
    # for now and normalize to (G and phi) % G.
    if not isinstance(node.op, ast.And):
      return node

    guards: list[ast.AST] = []
    values: list[ast.AST] = []
    changed = False

    for value in node.values:
      match value:
        case ast.BinOp(left=lhs, op=ast.Mod(), right=rhs):
          values.append(lhs)
          guards.append(rhs)
          changed = True
        case _:
          values.append(value)

    if not changed:
      return node

    out: ast.AST = ast.BoolOp(op=ast.And(), values=values)
    for guard in guards:
      out = ast.BinOp(left=out, op=ast.Mod(), right=guard)
    return out

  def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
    self.generic_visit(node)

    payload, guards = self._collect_guard_chain(node.body)
    if not guards:
      return node

    bound = {a.arg for a in node.args.args}
    bound.update(a.arg for a in node.args.kwonlyargs)
    if node.args.vararg is not None:
      bound.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
      bound.add(node.args.kwarg.arg)

    local_guards: list[ast.AST] = []
    hoisted_guards: list[ast.AST] = []
    for guard in guards:
      if _free_vars(guard) & bound:
        local_guards.append(guard)
      else:
        hoisted_guards.append(guard)

    if not hoisted_guards:
      return node

    inner_body = self._rebuild_guard_chain(payload, local_guards)
    out: ast.AST = ast.Lambda(args=node.args, body=inner_body)
    for guard in hoisted_guards:
      out = ast.BinOp(left=out, op=ast.Mod(), right=guard)
    return out

  # ---------- call rewrites and defined(...) folding ---------------
  def visit_Call(self, node: ast.Call) -> ast.AST:
    self.generic_visit(node)

    match node:
      case ast.Call(
        func=ast.Name(id="defined", ctx=ast.Load()),
        args=[ast.BinOp(op=ast.Mod(), right=rhs)],
        keywords=[],
      ):
        return rhs

      # Fold defined(expr) from static definedness, not just successful eval.
      case ast.Call(
        func=ast.Name(id="defined", ctx=ast.Load()),
        args=[arg],
        keywords=[],
      ):
        status = self._defined_status(arg)
        if status is not None:
          return ast.Constant(value=status)
        return node

      case ast.Call(
        func=ast.BinOp(left=lhs, op=ast.Mod(), right=rhs)
      ):
        node.func = lhs
        return ast.BinOp(left=node, op=ast.Mod(), right=rhs)

    guards: list[ast.AST] = []
    new_args: list[ast.AST] = []
    changed = False

    for arg in node.args:
      match arg:
        case ast.BinOp(left=lhs, op=ast.Mod(), right=rhs):
          new_args.append(lhs)
          guards.append(rhs)
          changed = True
        case _:
          new_args.append(arg)

    new_keywords: list[ast.keyword] = []
    for kw in node.keywords:
      if kw.arg is None:
        new_keywords.append(kw)
        continue

      match kw.value:
        case ast.BinOp(left=lhs, op=ast.Mod(), right=rhs):
          new_keywords.append(ast.keyword(arg=kw.arg, value=lhs))
          guards.append(rhs)
          changed = True
        case _:
          new_keywords.append(kw)

    if changed:
      node = ast.Call(func=node.func, args=new_args, keywords=new_keywords)
      for guard in guards:
        node = ast.BinOp(left=node, op=ast.Mod(), right=guard)
      return node

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

  TESTS = [
    ("(A % G) % G", "A % G"),
    ("A % (G1 % G2)", "A % G1 % G2"),
    ("(A % G1) % (G2 % G1)", "A % G1 % G2"),
  ]

  def _collect_chain(self, node: ast.AST) -> tuple[ast.AST, list[ast.AST]]:
    """Return (payload, guards) where guards are in application order."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mod):
      return node, []

    payload, guards = self._collect_chain(node.left)
    rhs_payload, rhs_guards = self._collect_chain(node.right)

    # right can itself be a guard-chain produced by earlier rewrites
    guards.append(rhs_payload)
    guards.extend(rhs_guards)
    return payload, guards

  @staticmethod
  def _rebuild_chain(payload: ast.AST, guards: list[ast.AST]) -> ast.AST:
    out = payload
    for guard in guards:
      out = ast.BinOp(left=out, op=ast.Mod(), right=guard)
    return out

  # ---------- core visitor ------------------------------------------
  def visit_BinOp(self, node: ast.BinOp):
    node = self.generic_visit(node)

    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)):
      return node

    payload, guards = self._collect_chain(node)

    seen: set[str] = set()
    unique_guards: list[ast.AST] = []
    for guard in guards:
      key = guard_key(guard)
      if key in seen:
        continue
      seen.add(key)
      unique_guards.append(guard)

    return self._rebuild_chain(payload, unique_guards)