# phosphorus/simplify/passes.py
from ast import *
from typing import List, Type
from .utils import is_literal
import ast


# ─────────────────────────────
#  Base class
# ─────────────────────────────
class SimplifyPass(NodeTransformer):
  """Base class: accepts an `env` mapping for transforms."""
  def __init__(self, env=None):
    super().__init__()
    self.env = {} if env is None else env


# ─────────────────────────────
#  Passes with Tests
# ─────────────────────────────
class NameInliner(SimplifyPass):
  """
  Inline identifiers whose run‑time value is a simple literal or has an .expr AST.
  """
  TEST_ENV = {"x":1}
  TESTS = [
    ("x", "1"),
    ("(1,2,3)[0]", "(1, 2, 3)[0]"),  # non‑literal index — no change
  ]
  def visit_Name(self, node: Name):
    if isinstance(node.ctx, Load) and node.id in self.env:
      val = self.env[node.id]
      if is_literal(val):
        return parse(repr(val), mode="eval").body
      # Inline any object with an .expr attribute that is an AST node
      if hasattr(val, "expr") and isinstance(val.expr, ast.AST):
        return val.expr
    return node


class DictMergeFolder(SimplifyPass):
  """
  Merge literal dicts: {'a':1} | {'b':2} → {'a':1, 'b':2}
  """
  TESTS = [
    ("{'a':1} | {'b':2}", "{'a': 1, 'b': 2}"),
    ("{'x':1} | {'x':2}", "{'x': 2}"),
  ]
  def visit_BinOp(self, node: BinOp):
    self.generic_visit(node)
    if isinstance(node.op, BitOr) and isinstance(node.left, Dict) and isinstance(node.right, Dict):
      merged = {}
      for d in (node.left, node.right):
        for k, v in zip(d.keys, d.values):
          if not isinstance(k, Constant):
            return node
          merged[k.value] = v
      return Dict(
        keys=[Constant(value=k) for k in merged],
        values=list(merged.values())
      )
    return node


class DictLookupFolder(SimplifyPass):
  """
  Fold dict lookups for literal keys or matching Name keys:
    {'a':1}['a'] → 1
    {i:1}[i]     → 1
  """
  TESTS = [
    ("{'a':1}['a']", "1"),
    ("{i:1}[i]", "1"),
  ]
  def visit_Subscript(self, node: Subscript):
    self.generic_visit(node)
    # constant key
    if isinstance(node.value, Dict) and isinstance(node.slice, Constant):
      mapping = {k.value: v for k, v in zip(node.value.keys, node.value.values)
                 if isinstance(k, Constant)}
      if node.slice.value in mapping:
        return mapping[node.slice.value]
    # Name key
    if isinstance(node.value, Dict) and isinstance(node.slice, Name):
      for k, v in zip(node.value.keys, node.value.values):
        if isinstance(k, Name) and k.id == node.slice.id:
          return v
    return node


class IfExpConstFolder(SimplifyPass):
  """
  Simplify conditional expressions when the test is Constant:
    a if True else b  → a
    a if False else b → b
  """
  TESTS = [
    ("a if True else b", "a"),
    ("a if False else b", "b"),
  ]
  def visit_IfExp(self, node: IfExp):
    self.generic_visit(node)
    if isinstance(node.test, Constant):
      return node.body if node.test.value else node.orelse
    return node


class BoolIdentityPruner(SimplifyPass):
  """
  Prune boolean identities:
    x and True   → x
    x and False  → False
    True and x   → x
    False and x  → False
    x or False   → x
    x or True    → True
    False or x   → x
    True or x    → True
  """
  TESTS = [
    ("x and True", "x"),
    ("x and False", "False"),
    ("True and x", "x"),
    ("False and x", "False"),
    ("x or False", "x"),
    ("x or True", "True"),
    ("False or x", "x"),
    ("True or x", "True"),
  ]
  def visit_BoolOp(self, node: BoolOp):
    self.generic_visit(node)
    match node:
      case BoolOp(op=And(), values=[Constant(value=True), rhs]):
        return rhs
      case BoolOp(op=And(), values=[lhs, Constant(value=True)]):
        return lhs
      case BoolOp(op=And(), values=[Constant(value=False), _]):
        return Constant(value=False)
      case BoolOp(op=And(), values=[_, Constant(value=False)]):
        return Constant(value=False)
      case BoolOp(op=Or(), values=[Constant(value=False), rhs]):
        return rhs
      case BoolOp(op=Or(), values=[lhs, Constant(value=False)]):
        return lhs
      case BoolOp(op=Or(), values=[Constant(value=True), _]):
        return Constant(value=True)
      case BoolOp(op=Or(), values=[_, Constant(value=True)]):
        return Constant(value=True)
    return node


class BoolConstFolder(SimplifyPass):
  """
  Fold boolean ops when *all* operands are Constant.
  """
  TESTS = [
    ("True and False", "False"),
    ("True or False", "True"),
    ("False and False", "False"),
  ]
  def visit_BoolOp(self, node: BoolOp):
    self.generic_visit(node)
    if all(isinstance(v, Constant) for v in node.values):
      vals = [v.value for v in node.values]
      if isinstance(node.op, And):
        return Constant(value=all(vals))
      if isinstance(node.op, Or):
        return Constant(value=any(vals))
    return node


# ─────────────────────────────
#  Pipeline registry
# ─────────────────────────────
PASS_PIPELINE: List[Type[SimplifyPass]] = [
  NameInliner,
  DictMergeFolder,
  DictLookupFolder,
  IfExpConstFolder,
  BoolIdentityPruner,
  BoolConstFolder,
]

from .lambda_pass import BetaReducer
PASS_PIPELINE.insert(1, BetaReducer)
from .macro_pass import MacroExpander
PASS_PIPELINE.append(MacroExpander)
from .guard_pass import GuardFolder
PASS_PIPELINE.append(GuardFolder)
