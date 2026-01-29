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
from typing import Any, Mapping

from .stypes import Type

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
  """Annotate nodes with .stype and .guard; strip DSL cues."""

  def __init__(self, env: Mapping[str, Any]):
    self.env: ChainMap[str, Any] = ChainMap({}, *([env] if env else []))

  # -------------------------------------------------------------------
  #  leaves
  # -------------------------------------------------------------------

  def visit_Name(self, node: ast.Name):
    val = self.env.get(node.id)
    if hasattr(val, "stype"):
      node.stype = val.stype
    if hasattr(val, "guard"):
      node.guard = val.guard
    return node

  def visit_Attribute(self, node: ast.Attribute):
    t = Type.from_spec(node.attr)
    if t is not None:
      node.value.stype = t
      return self.visit(node.value)          # strip .Type attribute
    node.value = self.visit(node.value)
    return node

  # -------------------------------------------------------------------
  #  DSL subscript [type]  or [type:guard]
  # -------------------------------------------------------------------

  def XXvisit_Subscript(self, node: ast.Subscript): #Turned off for now
    node.value = self.visit(node.value)
    type_spec, guard = _slice_to_spec(node.slice)

    if type_spec is not None:
      try:
        node.value.stype = Type.from_spec(type_spec)
      except Exception as e:
        LOG.warning("Invalid type spec %r: %s", type_spec, e)
        return node.value
      if guard is not None:
        node.value.guard = guard
      return node.value                     # strip DSL cue
    return super().generic_visit(node)

  # -------------------------------------------------------------------
  #  lambda  — add guard(lambda-expressed)
  # -------------------------------------------------------------------

  def visit_Lambda(self, node: ast.Lambda):
    # If the node already has a type (e.g., from a previous inference),
    # trust it and don't recompute
    if hasattr(node, "stype") and node.stype is not None:
      return node
    
    # 1. infer parameter types from default annotations
    rev_defaults = node.args.defaults[::-1]
    param_types = []
    for i, arg in enumerate(node.args.args[::-1]):
      dflt = rev_defaults[i] if i < len(rev_defaults) else None
      if isinstance(dflt, ast.Attribute) and isinstance(dflt.value, ast.Name) and dflt.value.id == "Type":
        param_types.append(Type.from_spec(dflt.attr) or Type.fresh())
      elif isinstance(dflt, ast.Name):
        param_types.append(Type.from_spec(dflt.id) or Type.fresh())
      else:
        param_types.append(Type.fresh())
    param_types.reverse()
    node.args.defaults = []

    # 2. recurse on body with params in env
    inner_env = {
      p.arg: type("_T", (), {"stype": t})() for p, t in zip(node.args.args, param_types)
    }
    body_node = self.__class__(self.env.new_child(inner_env)).visit(node.body)
    node.body = body_node

    # 3. function type:  param_types → body_t
    body_t = getattr(body_node, "stype", Type.fresh())
    fn_t = body_t
    for dom in reversed(param_types):
      fn_t = Type((dom, fn_t))
    node.stype = fn_t

    # 4. propagate guard: λx. BODY  →  guard := λx. BODY.guard
    body_guard = getattr(body_node, "guard", None)
    if body_guard is not None:
      node.guard = ast.Lambda(
        args=node.args, body=body_guard,
      )

    return node

  # -------------------------------------------------------------------
  #  calls  — compose guards
  #     TODO: infer types for atomic formulas: ALL_CAPS(x,y,...)
  # -------------------------------------------------------------------

  def visit_Call(self, node: ast.Call):
    node = self.generic_visit(node)

    # ---------- type inference (unchanged) ----------------------------
    fn_t = getattr(node.func, "stype", None)
    arg_t = getattr(node.args[0], "stype", None) if node.args else None
    if fn_t and fn_t.is_function:
      dom = fn_t.domain
      if arg_t and dom.is_unknown:
        fn_t = Type((arg_t, fn_t.range))
      elif arg_t and arg_t != dom:
        LOG.warning("Type mismatch: expected %s, got %s in %s",
                    fn_t.domain, arg_t, ast.unparse(node))
      node.stype = fn_t.range

    # ---------- guard propagation ------------------------------------
    fn_guard  = getattr(node.func, "guard", None)
    arg_guard = getattr(node.args[0], "guard", None) if node.args else None
    guards = []

    # apply function guard to argument
    if fn_guard is not None and node.args:
      guards.append(
        ast.Call(func=fn_guard, args=[node.args[0]], keywords=[])
      )

    if arg_guard is not None:
      guards.append(arg_guard)

    if guards:
      node.guard = guards[0] if len(guards) == 1 else \
        ast.BoolOp(ast.And(), guards)

    return node

  # -------------------------------------------------------------------
  #  binary operations - propagate type from left operand of %
  # -------------------------------------------------------------------

  def visit_BinOp(self, node: ast.BinOp):
    self.generic_visit(node)
    match node:
      case ast.BinOp(left=left, op=ast.Mod()):
        node.stype = getattr(left, "stype", None)
    return node

  # -------------------------------------------------------------------
  #  boolean operations - assume Type.t but log if operands mismatch
  # -------------------------------------------------------------------

  def visit_BoolOp(self, node: ast.BoolOp):
    """Handle boolean operators (And, Or)."""
    self.generic_visit(node)
    
    node.stype = Type.t
    
    # Check if any operand has a known type that's not Type.t or unknown
    for value in node.values:
      value_type = getattr(value, "stype", None)
      if value_type is not None and not value_type.is_unknown and value_type is not Type.t:
        LOG.warning("Type mismatch in boolean operation: expected %s, got %s in %s",
                   Type.t, value_type, ast.unparse(node))
    
    return node
  
  def visit_UnaryOp(self, node: ast.UnaryOp):
    """Handle unary operators, specifically Not."""
    self.generic_visit(node)
    
    if isinstance(node.op, ast.Not):
      node.stype = Type.t
      
      # Check if operand has a known type that's not Type.t or unknown
      operand_type = getattr(node.operand, "stype", None)
      if operand_type is not None and not operand_type.is_unknown and operand_type is not Type.t:
        LOG.warning("Type mismatch in Not operation: expected %s, got %s in %s",
                   Type.t, operand_type, ast.unparse(node))
    
    return node


# ---------------------------------------------------------------------------
#  public helper
# ---------------------------------------------------------------------------

def infer_and_strip(node: ast.AST, env: Mapping[str, Any] | None = None) -> ast.AST:
  """Annotate *node* with ``.stype`` and ``.guard``, strip DSL cues, and return the (possibly replaced) node."""
  return _Infer(env or {}).visit(node)


# ---------------------------------------------------------------------------
#  tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  import ast as _ast

  src1 = "(lambda x=Type.e: FLUFFY(x).t)(A)"
  tree1 = _ast.parse(src1, mode="eval").body
  node1 = infer_and_strip(tree1)
  assert getattr(node1, "stype", None) is Type.t

  src3 = "(lambda x=Type.e: x)(A)"
  tree3 = _ast.parse(src3, mode="eval").body
  node3 = infer_and_strip(tree3)
  assert getattr(node3, "stype", None) is Type.e

  src4 = "x[e]"
  tree4 = _ast.parse(src4, mode="eval").body
  node4 = infer_and_strip(tree4, {"x": type("_TypeHolder", (object,), {"stype": Type.fresh()})()})
  assert getattr(node4, "stype", None) is Type.e
  assert isinstance(node4, ast.Name)

  src5 = "x[e:guard]"
  tree5 = _ast.parse(src5, mode="eval").body
  node5 = infer_and_strip(tree5, {"x": type("_TypeHolder", (object,), {"stype": Type.fresh()})()})
  assert getattr(node5, "stype", None) is Type.e
  assert getattr(node5, "guard", None) is not None

  src6 = "x[(et, et_t)]"
  tree6 = _ast.parse(src6, mode="eval").body
  node6 = infer_and_strip(tree6, {"x": type("_TypeHolder", (object,), {"stype": Type.fresh()})()})
  assert getattr(node6, "stype", None) == Type((Type.et, Type.et_t))

  src7 = "x[(et, et_t):guard]"
  tree7 = _ast.parse(src7, mode="eval").body
  node7 = infer_and_strip(tree7, {"x": type("_TypeHolder", (object,), {"stype": Type.fresh()})()})
  assert getattr(node7, "stype", None) == Type((Type.et, Type.et_t))
  assert getattr(node7, "guard", None) is not None

  src8 = "x[e:e2:e3]"
  tree8 = _ast.parse(src8, mode="eval").body
  node8 = infer_and_strip(tree8, {"x": type("_TypeHolder", (object,), {"stype": Type.fresh()})()})
  assert getattr(node8, "stype", None) is Type.e  # "step" (e3) is ignored

  src9 = "(lambda x=Type.e: BARKS(x)[t:ANIMAL(x)])(A)"
  tree9 = _ast.parse(src9, mode="eval").body
  node9 = infer_and_strip(tree9)
  assert getattr(node9, "stype", None) is Type.t
  print(ast.unparse(node9.guard))

  print("✅ Extended subscript and lambda tests passed.")
