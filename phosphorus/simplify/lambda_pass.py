# phosphorus/simplify/lambda_passes.py
# -------------------------------------------------
# Lambda‑related optimisation passes for the Phosphorus
# expression simplifier.
# Provides a BetaReducer pass that performs β‑reduction with
# full, *local* α‑conversion via NameSubstituter.

from __future__ import annotations

from ast import *
from typing import Dict, Set
from copy import deepcopy

from .passes import SimplifyPass  # base class

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def free_vars(node: AST) -> Set[str]:
  """Collect free variable names inside *node*."""
  match node:
    case Name(id=name, ctx=Load()):
      return {name}
    case Lambda(args=args, body=body):
      bound = {a.arg for a in args.args}
      return free_vars(body) - bound
    case _:
      out: Set[str] = set()
      for child in iter_child_nodes(node):
        out.update(free_vars(child))
      return out


def _fresh(base: str, taken: Set[str]) -> str:
  """Generate a fresh identifier not in *taken*, based on *base*."""
  i = 0
  candidate = base
  while candidate in taken:
    i += 1
    candidate = f"{base}_{i}"
  return candidate

# ---------------------------------------------------------------------------
# Name substitution with built‑in α‑conversion
# ---------------------------------------------------------------------------
class _NameSubstituter(NodeTransformer):
  """
  Replace each occurrence of the names in *mapping* with the
  corresponding AST node **while** performing capture‑avoidance.

  When we enter a `Lambda`, parameters shadow outer names.  If a parameter
  collides with a name we would otherwise substitute or appears free in a
  replacement expression, we α‑rename that parameter to a fresh identifier
  and update the body accordingly.
  """
  def __init__(self, mapping: Dict[str, AST]):
    super().__init__()
    self.mapping = mapping

  def _alpha_and_recurse(self, node: Lambda) -> Lambda:
    bound = {a.arg for a in node.args.args}
    # If any parameter name is in mapping keys or appears free in replacement ASTs
    replacement_free = set().union(*(free_vars(v) for v in self.mapping.values()))
    collisions = bound & replacement_free  # only if replacement values free-vars collide with params

    if not collisions:
      # No renaming needed; just recurse, shadowing bound names
      inner_mapping = {k: v for k, v in self.mapping.items() if k not in bound}
      node.body = _NameSubstituter(inner_mapping).visit(node.body)
      return node

    # α‑rename colliding parameters
    taken = bound | free_vars(node.body) | set(self.mapping.keys())
    rename_map: Dict[str, str] = {}
    for old in collisions:
      new_name = _fresh(old, taken)
      rename_map[old] = new_name
      taken.add(new_name)

    # Apply renaming to parameters
    for arg in node.args.args:
      if arg.arg in rename_map:
        arg.arg = rename_map[arg.arg]

    # Substitute old param references inside body
    param_subst = {old: Name(id=new, ctx=Load()) for old, new in rename_map.items()}
    node.body = _NameSubstituter(param_subst).visit(node.body)

    # Recurse on remaining mapping
    inner_mapping = {k: v for k, v in self.mapping.items() if k not in collisions}
    node.body = _NameSubstituter(inner_mapping).visit(node.body)
    return node

  def visit_Lambda(self, node: Lambda) -> Lambda:
    # Deep-copy to avoid mutating original AST
    return self._alpha_and_recurse(deepcopy(node))

  def visit_Name(self, node: Name) -> AST:
    if isinstance(node.ctx, Load) and node.id in self.mapping:
      return deepcopy(self.mapping[node.id])
    return node

# ---------------------------------------------------------------------------
# Beta‑reducer pass
# ---------------------------------------------------------------------------
class BetaReducer(SimplifyPass):
  """Inline a lambda call with positional arguments, with capture‑avoidance."""

  TESTS = [
    ("(lambda x: x + 1)(3)",       "3 + 1"),
    ("((lambda x: (lambda y: x + y))(1))(2)", "1 + 2"),
    ("(lambda x, y: x * y)(2, 5)",  "2 * 5"),
    # capture‑avoidance of outer free variable
    ("(lambda y: (lambda x: x + y))(x)",    "lambda x_1: x_1 + x"),
    # capture‑avoidance inside inner lambda shadowing
    ("(lambda x: (lambda x: x + y))(3)",   "lambda x: x + y"),
  ]

  def visit_Call(self, node: Call) -> AST:
    self.generic_visit(node)
    if not isinstance(node.func, Lambda):
      return node

    params = [a.arg for a in node.func.args.args]
    if len(params) != len(node.args) or node.keywords:
      return node

    lam_copy: Lambda = deepcopy(node.func)
    subst_map: Dict[str, AST] = dict(zip(params, node.args))

    reducer = _NameSubstituter(subst_map)
    new_body = reducer.visit(lam_copy.body)
    return new_body
