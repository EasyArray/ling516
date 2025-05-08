# phosphorus/simplify/passes.py
from ast import *
from typing import List, Type
from collections import ChainMap
from .utils import is_literal
import ast

# Sentinel for shadowing the environment in SimplifyPass
_SHADOW = object()

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
#    TODO: inlining ASTs breaks alpha-conversion / acts as a macro
#          Is that what we want?
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

  def visit_Lambda(self, node: Lambda) -> Lambda:
    # collect all parameter names (positional, vararg, kwonly, kwarg)
    names = {a.arg for a in node.args.args}
    if node.args.vararg:   names.add(node.args.vararg.arg)
    for a in node.args.kwonlyargs: names.add(a.arg)
    if node.args.kwarg:    names.add(node.args.kwarg.arg)

    # push a new scope that shadows these names
    self.env = ChainMap({n: _SHADOW for n in names}, self.env)
    try:
      return self.generic_visit(node)
    finally:
      # pop that scope quickly: O(1)
      self.env = self.env.parents

  def visit_Name(self, node: Name):
    if isinstance(node.ctx, Load) and node.id in self.env:
      val = self.env[node.id]
      if val is _SHADOW:
        return node
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
    match(node):
      case BinOp(op=BitOr(), 
                 left=Dict(keys=keys1, values=values1), 
                 right=Dict(keys=keys2, values=values2)):
        # merge two dicts
        keys   = keys1 + keys2
        values = values1 + values2

        # use the source‐string of each key as the merge‐dict key
        merged: dict[str, tuple[AST, AST]] = {
          ast.unparse(k): (k, v)
          for k, v in zip(keys, values)
        }

        # rebuild our Dict node from the final (k,v) pairs
        if not merged:
          return node
        new_keys, new_values = zip(*merged.values())
        return Dict(keys=list(new_keys), values=list(new_values))

    return node


class DictLookupFolder(SimplifyPass):
  """
  Fold dict lookups for literal keys or matching Name keys:
    {'a':1}['a'] → 1
    {i:1}[i]     → 1
    (d | {1:x})[1] → x
  """
  TESTS = [
    ("{'a':1}['a']", "1"),
    ("{i:1}[i]", "1"),
    ("(d | {1:x})[1]", "x"),
  ]
  def visit_Subscript(self, node: Subscript):
    self.generic_visit(node)
    match node:
      # {'a':1}['a'] → 1 (constant key)
      case Subscript(value=Dict(keys=keys, values=values), slice=Constant(value=key)):
        mapping = {k.value: v for k, v in zip(keys, values) if isinstance(k, Constant)}
        if key in mapping:
          return mapping[key]
      # {i:1}[i] → 1 (name key)
      case Subscript(value=Dict(keys=keys, values=values), slice=Name(id=key_id)):
        for k, v in zip(keys, values):
          if isinstance(k, Name) and k.id == key_id:
            return v
      # (d | {k: v})[k] → v (RHS dict, constant key)
      case Subscript(
        value=BinOp(op=BitOr(), right=Dict(keys=rhs_keys, values=rhs_values)),
        slice=Constant(value=key)
      ):
        for k, v in zip(rhs_keys, rhs_values):
          if isinstance(k, Constant) and k.value == key:
            return v
      # (d | {k: v})[k] → v (RHS dict, name key)
      case Subscript(
        value=BinOp(op=BitOr(), right=Dict(keys=rhs_keys, values=rhs_values)),
        slice=Name(id=key_id)
      ):
        for k, v in zip(rhs_keys, rhs_values):
          if isinstance(k, Name) and k.id == key_id:
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
from .guard_pass import GuardFolder, RemoveDuplicateGuards
PASS_PIPELINE.append(GuardFolder)
PASS_PIPELINE.append(RemoveDuplicateGuards)
