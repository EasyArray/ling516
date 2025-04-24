"""phosphorus.core.infer
---------------------------------
Side‑effecting type inference for the mini‑DSL.

* Walks a **pre‑simplification** AST.
* Strips DSL cues so the tree that reaches the runtime contains **no**
  ``Type.*`` references or ``.t``/``.et`` attributes.
* Current cues:

  * **Parameter defaults**  – long form ``=Type.e`` *or* shorthand
    ``=e``, ``=et``, …   Examples::

        lambda x=Type.e: ...
        lambda f=eet: ...

  * **Body attribute**      – ``expr.t``, ``expr.e`` … marks the return
    type of the surrounding lambda.

* Computes a best‑effort :class:`~phosphorus.core.stypes.Type`; returns
  ``None`` when unable.

Extend this module when you add more DSL features (logic predicates,
polymorphism, …).
"""

from __future__ import annotations

import ast
import logging
from collections import ChainMap
from typing import Any, Mapping, Optional

from phosphorus.core import stypes
from phosphorus.core.phivalue import PhiValue

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _stype_from_token(tok: str) -> Optional[stypes.Type]:
  """Translate ``'e'``, ``'t'``, ``'et'`` … → cached :class:`Type`."""
  try:
    return getattr(stypes.Type, tok)
  except AttributeError:
    return None


# ---------------------------------------------------------------------------
#  main transformer / visitor
# ---------------------------------------------------------------------------

class _Infer(ast.NodeTransformer):
  """Mutates *node* in‑place and returns a :class:`Type` or ``None``."""

  def __init__(self, env: Mapping[str, Any]):
    # ChainMap lets us cheaply push/pull lambda scopes
    self.env: ChainMap[str, Any] = ChainMap({}, *([env] if env else []))

  # -------------------------------------------------------------------
  #  leaves
  # -------------------------------------------------------------------

  def visit_Name(self, node: ast.Name):
    val = self.env.get(node.id)
    if isinstance(val, PhiValue):
      return node, val.stype
    if hasattr(val, "stype"):
      return node, val.stype
    return node, None

  def visit_Attribute(self, node: ast.Attribute):
    t = _stype_from_token(node.attr)
    if t is not None:
      # strip any .t/.e/.et etc. suffix and record return type
      return node.value, t
    # otherwise recurse into the base expression to clean nested cues
    new_val, _ = self.visit(node.value)
    node.value = new_val
    return node, None

  # -------------------------------------------------------------------
  #  lambda
  # -------------------------------------------------------------------

  def visit_Lambda(self, node: ast.Lambda):
    # Build map param -> default expression (defaults align from the end)
    rev_defaults = list(node.args.defaults)[::-1]
    param_defaults = {
      arg.arg: rev_defaults[i] if i < len(rev_defaults) else None
      for i, arg in enumerate(node.args.args[::-1])
    }

    # Infer parameter types, strip defaults
    param_types: list[stypes.Type] = []
    for arg in node.args.args:
      dflt = param_defaults[arg.arg]
      if (
        isinstance(dflt, ast.Attribute)
        and isinstance(dflt.value, ast.Name)
        and dflt.value.id == "Type"
      ):
        t = _stype_from_token(dflt.attr) or stypes.Type.fresh()
      elif isinstance(dflt, ast.Name):
        t = _stype_from_token(dflt.id) or stypes.Type.fresh()
      else:
        t = stypes.Type.fresh()
      param_types.append(t)
    node.args.defaults = []  # remove defaults

    # Infer body with a fresh transformer using extended env
    inner_env = {
      p.arg: PhiValue(ast.Name(id=p.arg, ctx=ast.Load()), stype=p_t)
      for p, p_t in zip(node.args.args, param_types)
    }
    infer_inner = _Infer(self.env.new_child(inner_env))
    new_body, body_t = infer_inner.visit(node.body)
    node.body = new_body
    # Default to fresh type if none inferred
    if body_t is None:
      body_t = stypes.Type.fresh()

    # Build function type: param1 -> (param2 -> ... -> body_t)
    fn_t = body_t
    for dom in reversed(param_types):
      fn_t = stypes.Type((dom, fn_t))
    return node, fn_t

  # -------------------------------------------------------------------
  #  calls
  # -------------------------------------------------------------------

  def visit_Call(self, node: ast.Call):
    new_func, fn_t = self.visit(node.func)
    node.func = new_func

    if node.args:
      new_arg, arg_t = self.visit(node.args[0])
      node.args[0] = new_arg
    else:
      arg_t = None

    if fn_t and fn_t.is_function:
      if arg_t and arg_t != fn_t.domain:
        LOG.warning("Type mismatch: expected %s, got %s in %s", fn_t.domain, arg_t, ast.unparse(node))
      return node, fn_t.range
    return node, None

  # -------------------------------------------------------------------
  #  generic (preserve tuple contract)
  # -------------------------------------------------------------------

  def generic_visit(self, node):  # noqa: D401
    base = super().generic_visit(node)
    return (base, None) if not isinstance(base, tuple) else base


# ---------------------------------------------------------------------------
#  public helper
# ---------------------------------------------------------------------------

def infer_type(node: ast.AST, env: Mapping[str, Any] | None = None):
  """Strip DSL artefacts from *node* and return its :class:`Type`."""
  stripped, t = _Infer(env or {}).visit(node)
  if stripped is not node:
    node.__dict__.update(stripped.__dict__)
  return t


# ---------------------------------------------------------------------------
#  tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  import ast as _ast

  # long‑form default + .t attribute
  src1 = "(lambda x=Type.e: FLUFFY(x).t)(A)"
  tree1 = _ast.parse(src1, mode="eval").body
  assert infer_type(tree1) is stypes.t
  assert "Type.e" not in ast.unparse(tree1) and ".t" not in ast.unparse(tree1)

  # shorthand defaults  =e   and   =eet
  src2 = "(lambda x=e: (lambda f=eet: f(x))(G))(A)"
  tree2 = _ast.parse(src2, mode="eval").body
  t2 = infer_type(tree2)
  assert t2 is not None  # should infer some type chain

  # Test 3: identity lambda with Type.e default yields E
  src3 = "(lambda x=Type.e: x)(A)"
  tree3 = _ast.parse(src3, mode="eval").body
  assert infer_type(tree3) is stypes.e
  assert "Type.e" not in ast.unparse(tree3)

  print("✅ infer_type extended tests passed.")
