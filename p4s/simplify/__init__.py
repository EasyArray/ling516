# phosphorus/simplify/__init__.py
from .passes import PASS_PIPELINE
from .utils  import capture_env
import ast
from typing import Union

__all__ = ["simplify", "run_pass_tests"]


def simplify(expr: Union[str, ast.AST], *, max_iter: int = 5, env: dict | None = None) -> ast.AST:
  """
  Simplify a Python expression, given as a source string or AST, and
  return the simplified AST. Uses PASS_PIPELINE to transform the AST
  until it stabilizes or max_iter is reached. Optional env override.
  """
  # prepare environment
  if env is None:
    env = capture_env()

  # parse or accept AST
  if isinstance(expr, str):
    expr_ast = ast.parse(expr, mode="eval").body  # type: ignore[assignment]
  elif isinstance(expr, ast.AST):
    expr_ast = expr
  else:
    raise TypeError("simplify() expects a source-code string or ast.AST")

  # fixed-point iteration over passes
  for _ in range(max_iter):
    old_dump = ast.dump(expr_ast, annotate_fields=False)
    for pass_cls in PASS_PIPELINE:
      expr_ast = pass_cls(env).visit(expr_ast)
    ast.fix_missing_locations(expr_ast)
    if ast.dump(expr_ast, annotate_fields=False) == old_dump:
      break
  else:
    raise RuntimeError("simplify() did not converge within max_iter passes")

  return expr_ast


def run_pass_tests():
  """
  Run TESTS for each pass through the full pipeline.
  Each pass class may define:
    - TEST_ENV: dict for name inlining or macros
    - TESTS: list of (src, expected_src)
  Outputs comparison of ast.unparse(simplified AST) vs expected.
  """
  from ast import parse, unparse

  # per-pass tests
  for cls in PASS_PIPELINE:
    tests = getattr(cls, 'TESTS', None)
    if not tests:
      continue
    env_override = getattr(cls, 'TEST_ENV', {}) or {}
    print(f"== Pipeline tests for {cls.__name__} ==")
    for src, expected in tests:
      ast_out = simplify(src, env=env_override)
      out = unparse(ast_out)
      status = 'OK' if out == expected else f"FAIL (got {out!r})"
      print(f"{src!r} -> {out!r}    [{status}]")
    print()

  # integration tests, if present
  INTEGRATION_TESTS = globals().get('INTEGRATION_TESTS')
  if INTEGRATION_TESTS:
    print("== Integration tests ==")
    for src, expected in INTEGRATION_TESTS:
      ast_out = simplify(src, env={})
      out = unparse(ast_out)
      status = 'OK' if out == expected else f"FAIL (got {out!r})"
      print(f"{src!r} -> {out!r}    [{status}]")

if __name__ == "__main__":
  run_pass_tests()