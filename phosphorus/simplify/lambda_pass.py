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
  """Inline lambda calls (positional *and* keyword) and substitute free
  variables via keyword arguments. For non‑lambda callables we perform the
  substitution only when the call has **no positional arguments**—this
  mirrors ``PhiValue.__call__`` where keywords are treated as environment
  overrides.
  """

  TESTS = [
    # positional + keyword on lambda
    ("(lambda x, y: x * y)(2, y=5)",               "2 * 5"),
    # free‑var binding via keyword
    ("(lambda x: LOVES(x,a))(Mary, a=John)",        "LOVES(Mary, John)"),
    # non‑lambda callable with keyword‑only override
    ("LOVES(Mary,a)(a=John)",                       "LOVES(Mary, John)"),
    # original positional‑only inlining
    ("(lambda x, y: x * y)(2, 5)",                 "2 * 5"),
    # nested lambdas with positional args
    ("((lambda x: (lambda y: x + y))(1))(2)",      "1 + 2"),
    # capture‑avoidance with keyword binding
    ("(lambda y: (lambda x: x + y))(y=x)",          "lambda x_1: x_1 + x"),
  ]

  # ------------------------------------------------------------------
  # main visitor
  # ------------------------------------------------------------------
  def visit_Call(self, node: Call) -> AST:
    # First recurse into children
    self.generic_visit(node)

    # Fast‑path: no keyword args → original positional‑only logic
    if not node.keywords:
      return self._inline_lambda_positional_only(node)

    # Ignore calls with **kwargs (kw.arg is None)
    if any(kw.arg is None for kw in node.keywords):
      return node

    kw_bindings: dict[str, AST] = {kw.arg: kw.value for kw in node.keywords}

    # ------------------------------------------------------------------
    # CASE 1 : literal Lambda target
    # ------------------------------------------------------------------
    if isinstance(node.func, Lambda):
      lam: Lambda = node.func
      params = [p.arg for p in lam.args.args]

      # Too many positional args → leave untouched
      if len(node.args) > len(params):
        return node

      # 1) bind parameters from positional args
      param_bindings: dict[str, AST] = dict(zip(params, node.args))

      # 2) bind parameters from keyword args (error on duplicates)
      for name, value in kw_bindings.items():
        if name in params:
          if name in param_bindings:
            return node  # duplicate binding
          param_bindings[name] = value

      # Abort if any parameter unbound
      if len(param_bindings) != len(params):
        return node

      # Unified substitution map: parameters + extra keyword env
      subst_map = param_bindings | {k: v for k, v in kw_bindings.items() if k not in params}

      body_copy = deepcopy(lam.body)
      new_body = _NameSubstituter(subst_map).visit(body_copy)
      return new_body

    # ------------------------------------------------------------------
    # CASE 2 : arbitrary callable (e.g. LOVES(Mary,a)(a=John))
    # Only when there are *no* positional args.
    # ------------------------------------------------------------------
    if node.args:
      return node  # positional args present → leave untouched

    func_copy = deepcopy(node.func)
    func_copy = _NameSubstituter(kw_bindings).visit(func_copy)
    return func_copy
  
  # ------------------------------------------------------------------
  # helper: original positional‑only lambda inliner (unchanged)
  # ------------------------------------------------------------------
  def _inline_lambda_positional_only(self, node: Call) -> AST:
    if not isinstance(node.func, Lambda):
      return node

    lam: Lambda = node.func
    params = [p.arg for p in lam.args.args]
    if len(params) != len(node.args):
      return node

    lam_copy = deepcopy(lam)
    subst_map = dict(zip(params, node.args))
    return _NameSubstituter(subst_map).visit(lam_copy.body)