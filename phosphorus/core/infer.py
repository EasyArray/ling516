"""phosphorus.core.infer – rewrite

*   Instead of returning (node, type) tuples, we annotate every visited
    *expression* node with a ``stype`` attribute.  That avoids inserting
    tuples into the AST and eliminates the unparser KeyError.
*   The transformer still strips DSL cues in‑place.
*   ``infer_type`` now simply returns ``getattr(node, 'stype', None)``
    after walking.
"""

from __future__ import annotations

import ast
import logging
from collections import ChainMap
from typing import Any, Mapping, Optional

from phosphorus.core import stypes

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _stype_from_token(tok: str) -> Optional[stypes.Type]:
  try:
    return getattr(stypes.Type, tok)
  except AttributeError:
    return None


# ---------------------------------------------------------------------------
#  main transformer (adds .stype attribute)
# ---------------------------------------------------------------------------

class _Infer(ast.NodeTransformer):
  """Annotates expression nodes with ``.stype``; strips DSL cues."""

  def __init__(self, env: Mapping[str, Any]):
    self.env: ChainMap[str, Any] = ChainMap({}, *([env] if env else []))

  # -------------------------------------------------------------------
  #  leaves
  # -------------------------------------------------------------------

  def visit_Name(self, node: ast.Name):
    val = self.env.get(node.id)
    if hasattr(val, "stype"):
      node.stype = val.stype
    return node

  def visit_Attribute(self, node: ast.Attribute):
    t = _stype_from_token(node.attr)
    if t is not None:
      node.value.stype = t
      return self.visit(node.value)  # strip attribute
    node.value = self.visit(node.value)
    return node

  # -------------------------------------------------------------------
  #  lambda
  # -------------------------------------------------------------------

  def visit_Lambda(self, node: ast.Lambda):
    from phosphorus.core.phivalue import PhiValue

    rev_defaults = list(node.args.defaults)[::-1]
    param_defaults = {
      arg.arg: rev_defaults[i] if i < len(rev_defaults) else None
      for i, arg in enumerate(node.args.args[::-1])
    }

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
    node.args.defaults = []

    inner_env = {
      p.arg: PhiValue(ast.Name(id=p.arg, ctx=ast.Load()), stype=t)
      for p, t in zip(node.args.args, param_types)
    }
    body_node = self.__class__(self.env.new_child(inner_env)).visit(node.body)
    node.body = body_node
    body_t = getattr(body_node, "stype", stypes.Type.fresh())

    fn_t = body_t
    for dom in reversed(param_types):
      fn_t = stypes.Type((dom, fn_t))
    node.stype = fn_t
    return node

  # -------------------------------------------------------------------
  #  calls
  # -------------------------------------------------------------------

  def visit_Call(self, node: ast.Call):
    node.func = self.visit(node.func)
    if node.args:
      node.args[0] = self.visit(node.args[0])

    fn_t  = getattr(node.func, "stype", None)
    arg_t = getattr(node.args[0], "stype", None) if node.args else None

    if fn_t and fn_t.is_function:
      if arg_t and arg_t != fn_t.domain:
        LOG.warning("Type mismatch: expected %s, got %s in %s", fn_t.domain, arg_t, ast.unparse(node))
      node.stype = fn_t.range
    return node

  # -------------------------------------------------------------------
  #  generic
  # -------------------------------------------------------------------

  def generic_visit(self, node):  # noqa: D401
    return super().generic_visit(node)


# ---------------------------------------------------------------------------
#  public helper
# ---------------------------------------------------------------------------

def infer_type(node: ast.AST, env: Mapping[str, Any] | None = None):
  """Annotate *node* with ``.stype`` and return it (may be ``None``)."""
  _Infer(env or {}).visit(node)
  return getattr(node, "stype", None)


# ---------------------------------------------------------------------------
#  tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  import ast as _ast

  src1 = "(lambda x=Type.e: FLUFFY(x).t)(A)"
  tree1 = _ast.parse(src1, mode="eval").body
  assert infer_type(tree1) is stypes.t

  src3 = "(lambda x=Type.e: x)(A)"
  tree3 = _ast.parse(src3, mode="eval").body
  assert infer_type(tree3) is stypes.e

  print("✅ infer_type attribute tests passed.")
  