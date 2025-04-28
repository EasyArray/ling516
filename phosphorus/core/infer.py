"""phosphorus.core.infer 
*   Annotate every visited
    *expression* node with a ``stype`` attribute.  That avoids inserting
    tuples into the AST and eliminates the unparser KeyError.
*   The transformer strips DSL cues in‑place.
*   ``infer_type`` now simply returns ``getattr(node, 'stype', None)``
    after walking.
"""

from __future__ import annotations

import ast
import logging
from collections import ChainMap
from typing import Any, Mapping, Optional

from phosphorus.core.stypes import Type

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  helper to parse a slice into a type spec and optional guard
# ---------------------------------------------------------------------------

def _slice_to_spec(slice_node: ast.AST) -> tuple:
  """Convert a subscript slice into a type spec (string or nested tuple)
     or a (type_spec, guard_ast) pair"""
  match slice_node:
    case ast.Name(id=tok):
      return tok, None
    case ast.Tuple(elts=elts):
      return tuple(_slice_to_spec(e)[0] for e in elts), None
    case ast.Slice(lower=low, upper=guard):
      # [type:guard] or [ (tuple):guard ]
      typ_spec, _ = _slice_to_spec(low)
      return (typ_spec, guard)
    case _:
      return None, None

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
    t = Type.from_spec(node.attr)
    if t is not None:
      node.value.stype = t
      return self.visit(node.value)  # strip attribute
    node.value = self.visit(node.value)
    return node

  def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
    """Support DSL: expr[type] and expr[type:guard] annotations."""
    # Recurse into the base expression
    node.value = self.visit(node.value)

    # Parse the slice into a spec (type_spec, optional_guard)
    type_spec, guard = _slice_to_spec(node.slice)
    if type_spec is not None:
      # Build and assign the type, warning on invalid specs
      try:
        t = Type.from_spec(type_spec)
        node.value.stype = t
      except Exception as e:
        LOG.warning("Invalid type spec %r: %s", type_spec, e)
        return node.value

      # Attach guard if provided
      if guard is not None:
        node.value.guard = guard
      return node.value

    # fallback for non‑DSL subscripts
    return super().generic_visit(node)

  # -------------------------------------------------------------------
  #  lambda
  # -------------------------------------------------------------------

  def visit_Lambda(self, node: ast.Lambda):
    rev_defaults = list(node.args.defaults)[::-1]
    param_defaults = {
      arg.arg: rev_defaults[i] if i < len(rev_defaults) else None
      for i, arg in enumerate(node.args.args[::-1])
    }

    param_types: list[Type] = []
    for arg in node.args.args:
      dflt = param_defaults[arg.arg]
      if (
        isinstance(dflt, ast.Attribute)
        and isinstance(dflt.value, ast.Name)
        and dflt.value.id == "Type"
      ):
        t = Type.from_spec(dflt.attr) or Type.fresh()
      elif isinstance(dflt, ast.Name):
        t = Type.from_spec(dflt.id) or Type.fresh()
      else:
        t = Type.fresh()
      param_types.append(t)
    node.args.defaults = []

    inner_env = {
      p.arg: type("_TypeHolder", (object,), {"stype": t})()
      for p, t in zip(node.args.args, param_types)
    }
    body_node = self.__class__(self.env.new_child(inner_env)).visit(node.body)
    node.body = body_node
    body_t = getattr(body_node, "stype", Type.fresh())

    fn_t = body_t
    for dom in reversed(param_types):
      fn_t = Type((dom, fn_t))
    node.stype = fn_t
    return node

  # -------------------------------------------------------------------
  #  calls
  # -------------------------------------------------------------------

  def visit_Call(self, node: ast.Call):
    node.func = self.visit(node.func)
    if node.args:
      node.args[0] = self.visit(node.args[0])

    fn_t = getattr(node.func, "stype", None)
    arg_t = getattr(node.args[0], "stype", None) if node.args else None

    if fn_t and fn_t.is_function:
      dom = fn_t.domain
      if arg_t and dom.is_unknown:
        fn_t = Type((arg_t, fn_t.range))
      elif arg_t and arg_t != dom:
        LOG.warning("Type mismatch: expected %s, got %s in %s", fn_t.domain, arg_t, ast.unparse(node))
      node.stype = fn_t.range
    return node



# ---------------------------------------------------------------------------
#  public helper
# ---------------------------------------------------------------------------

def infer_type(node: ast.AST, env: Mapping[str, Any] | None = None):
  """Annotate *node* with ``.stype`` and ``.guard`` and return them (may be ``None``)."""
  node = _Infer(env or {}).visit(node)
  return getattr(node, "stype", None), getattr(node, "guard", None)


# ---------------------------------------------------------------------------
#  tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  import ast as _ast

  src1 = "(lambda x=Type.e: FLUFFY(x).t)(A)"
  tree1 = _ast.parse(src1, mode="eval").body
  type1, _ = infer_type(tree1)
  assert type1 is Type.t

  src3 = "(lambda x=Type.e: x)(A)"
  tree3 = _ast.parse(src3, mode="eval").body
  type3, _ = infer_type(tree3)
  assert type3 is Type.e

  src4 = "x[e]"
  tree4 = _ast.parse(src4, mode="eval").body
  type4, _ = infer_type(tree4, {"x": type("_TypeHolder", (object,), {"stype": Type.fresh()})()})
  assert type4 is Type.e

  src5 = "x[e:guard]"
  tree5 = _ast.parse(src5, mode="eval").body
  type5, guard5 = infer_type(tree5, {"x": type("_TypeHolder", (object,), {"stype": Type.fresh()})()})
  assert type5 is Type.e
  assert guard5 is not None

  src6 = "x[(et, et_t)]"
  tree6 = _ast.parse(src6, mode="eval").body
  type6, _ = infer_type(tree6, {"x": type("_TypeHolder", (object,), {"stype": Type.fresh()})()})
  assert type6 == Type((Type.et, Type.et_t))

  src7 = "x[(et, et_t):guard]"
  tree7 = _ast.parse(src7, mode="eval").body
  type7, guard7 = infer_type(tree7, {"x": type("_TypeHolder", (object,), {"stype": Type.fresh()})()})
  assert type7 == Type((Type.et, Type.et_t))
  assert guard7 is not None

  src8 = "x[e:e2:e3]"
  tree8 = _ast.parse(src8, mode="eval").body
  type8, _ = infer_type(tree8, {"x": type("_TypeHolder", (object,), {"stype": Type.fresh()})()})
  assert type8 is Type.e  # "step" (e3) is ignored

  src9 = "(lambda x=Type.e: FLUFFY(x)[t])(A)"
  tree9 = _ast.parse(src1, mode="eval").body
  type9, _ = infer_type(tree1)
  assert type9 is Type.t

  print("✅ Extended subscript and lambda tests passed.")
