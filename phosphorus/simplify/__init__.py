# phosphorus/simplify/__init__.py
from .passes import PASS_PIPELINE
from .utils  import capture_env
import ast

__all__ = ["simplify", "run_pass_tests"]


def simplify(src: str, *, max_iter: int = 5, env: dict | None = None) -> str:
  """
  Return a semantically‑equivalent but syntactically simpler
  expression string. Runs PASS_PIPELINE repeatedly until the AST stops
  changing or `max_iter` iterations have passed.
  Allows optional `env` override for testing or custom environments.
  """
  if not isinstance(src, str):
    raise TypeError("simplify() expects a source-code string")

  # capture or use provided environment
  if env is None:
    env = capture_env()

  # parse into AST
  expr_ast = ast.parse(src, mode="eval").body

  # fixed-point iteration
  for _ in range(max_iter):
    old_dump = ast.dump(expr_ast, annotate_fields=False)
    for pass_cls in PASS_PIPELINE:
      expr_ast = pass_cls(env).visit(expr_ast)
    ast.fix_missing_locations(expr_ast)

    if ast.dump(expr_ast, annotate_fields=False) == old_dump:
      break
  else:
    raise RuntimeError("simplify() did not converge within max_iter passes")

  return ast.unparse(expr_ast)

# ── integration tests ──
INTEGRATION_TESTS = [
  # merge then lookup
  ("({'a':1} | {'b':2})['a']",   "1"),
  # nested if + bool pruning (assuming a=4 in captured env)
  ("(a if True else b) and True", "4"),
  # lookup with name inlining then bool-fold
  ("({i:1}[i] or False)",         "1"),
]
INTEGRATION_ENV = {
  "a": 4,
}

def run_pass_tests():
  """
  Run all TESTS of each pass through the full pipeline.
  Each pass class may define:
    - TEST_ENV: dict for name inlining
    - TESTS: list of (src, expected_src)
  """
  from ast import parse, unparse
  for cls in PASS_PIPELINE:
    tests = getattr(cls, 'TESTS', None)
    if not tests:
      continue
    env_override = getattr(cls, 'TEST_ENV', {}) or {}
    print(f"== Pipeline tests for {cls.__name__} ==")
    for src, expected in tests:
      out = simplify(src, env=env_override)
      status = 'OK' if out == expected else f"FAIL (got {out!r})"
      print(f"{src!r} -> {out!r}    [{status}]")
    print()

  print("== Integration tests ==")
  for src, expected in INTEGRATION_TESTS:
    out = simplify(src, env=INTEGRATION_ENV)
    status = "OK" if out == expected else f"FAIL (got {out!r})"
    print(f"{src!r} -> {out!r}    [{status}]")
