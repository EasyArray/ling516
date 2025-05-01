'''Constants used throughout the Phosphorus project.
'''

class _Sentinel:
  '''A sentinel class for unique constant values.'''
  def __init__(self, name):
    self.name = name
  def __repr__(self):
    return f"<{self.name}>"
  def __str__(self):
    return self.name
  def _repr_html_(self):
    return f"<code>&lt;{self.name}&gt;</code>"

UNDEF = _Sentinel("UNDEF")
VACUOUS = _Sentinel("VACUOUS")
