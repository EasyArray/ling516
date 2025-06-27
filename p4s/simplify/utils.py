# phosphorus/simplify/utils.py
import inspect
from collections import ChainMap


def capture_env(skip: int = 0):
  """
  Walk the call stack, skipping `skip` frames, and build a ChainMap of:
    • all f_locals (from inner to outer),
    • then the globals of the topmost frame.

  `skip` determines how many additional frames beyond this function
  to omit (e.g., use skip=1 to start at the caller’s caller).
  """
  frame = inspect.currentframe()
  # advance past this frame plus any skipped frames
  for _ in range(skip + 1):
    if frame is None:
      break
    frame = frame.f_back

  maps = []
  last_globals = {}
  while frame:
    maps.append(frame.f_locals)
    last_globals = frame.f_globals
    frame = frame.f_back

  return ChainMap(*maps, last_globals)


def is_literal(val) -> bool:
  """
  Return True if `val` is a Python literal we can inline safely:
    • primitives: str, bytes, bool, int, float, None
    • nested tuples and lists of literals
  """
  from ast import Constant

  if isinstance(val, (str, bytes, bool, int, float, type(None))):
    return True
  if isinstance(val, (tuple, list)):
    return all(is_literal(item) for item in val)
  # you could extend to dict/frozenset here if desired
  return False
