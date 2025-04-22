import inspect
from collections import ChainMap

def capture_env():
  """
  Walk up the call stack and build a ChainMap of
  all f_locals (inner → outer), then f_globals.
  """
  frame = inspect.currentframe().f_back
  maps = []
  while frame:
    maps.append(frame.f_locals)
    frame = frame.f_back
  # last frame’s globals:
  globals_ = maps[-1].get('__builtins__', {})  # or frame.f_globals
  return ChainMap(*maps, globals_)

def is_literal(val) -> bool:
  """
  Return True if `val` is a Python literal we can inline safely.
  Accept simple immutables and recursively on tuples/lists.
  """
  from ast import Constant

  if isinstance(val, (str, bytes, bool, int, float, type(None))):
    return True
  if isinstance(val, (tuple, list)):
    return all(is_literal(item) for item in val)
  # you could extend to dict/frozenset here if desired
  return False