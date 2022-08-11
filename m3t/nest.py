import types

import numpy

try:
  import tensorflow

  # Create canonical names for tf.nest functions
  try:
    _tf_nest = tensorflow.nest
  except AttributeError:
    _tf_nest = tensorflow.contrib.framework.nest

  tf = types.SimpleNamespace()
  tf.map_structure = _tf_nest.map_structure
except ImportError:
  pass


def _np_map_structure( f, struct ):
  """ Returns an object with the same structure as 'struct' where each leaf
  element 'x' is replaced with 'f(x)'.
  """
  if hasattr(struct, "_fields"): # namedtuple
    return type(struct)( *(_np_map_structure(f, e) for e in struct) )
  elif isinstance(struct, (tuple, list)):
    return type(struct)( _np_map_structure(f, e) for e in struct )
  else:
    return f( struct )
    
def _np_iter_flatten( struct ):
  """ Returns a generator that yields leaf elements from 'struct' in a
  fixed order.
  """
  def f( struct ):
    if hasattr(struct, "_fields"): # namedtuple
      for e in struct:
        yield from f(e)
    elif isinstance(struct, (tuple, list)):
      for e in struct:
        yield from f(e)
    else:
      yield struct
  yield from f( struct )
    
def _np_flatten( struct ):
  """ Returns a list containing the leaf elements of 'struct' in a
  fixed order.
  """
  return list(_np_iter_flatten( struct ))
  
def _np_pack_sequence_as( struct, flat_sequence ):
  """ Returns an object with the same structure as 'struct' where the leaf
  elements are taken from 'flat_sequence'. Satisfies:
    ```struct == pack_sequence_as( struct, flatten( struct ) )```
  """
  itr = iter(flat_sequence)
  def f( x ):
    return next(itr)
  return _np_map_structure( f, struct )
  
def _np_zip_with( f, nests ):
  """ Combines a sequence of nests using the supplied function.
  
  Given a list of nests with the same structure, returns a nest that also
  shares the same structure where each leaf is the result of applying 
  f( [x1, x2, ...] ) to the list of corresponding leaf Tensors from each nest.
  """
  if not nests:
    return nests
    
  # for e0, e1 in zip(nests, nests[1:]):
    # _np_assert_same_structure( e0, e1 )
    
  flat = [_np_flatten( e ) for e in nests]
  zipped = []
  for i in range(len(flat[0])):
    zipped.append( f( [e[i] for e in flat] ) )
  return _np_pack_sequence_as( nests[0], zipped )

  
np = types.SimpleNamespace()
np.flatten = _np_flatten
np.iter_flatten = _np_iter_flatten
np.map_structure = _np_map_structure
np.pack_sequence_as = _np_pack_sequence_as
np.zip_with = _np_zip_with

# ----------------------------------------------------------------------------

def test():
  x = (
    (1, 2, ()),
    [[], (), 3],
    4
  )
  
  y = (
    ("a", "b", ()),
    [[], (), "c"],
    "d"
  )
  
  z = (
    (6, 7, ()),
    [[], (), 8],
    9
  )
  
  print( x )
  
  flat = _np_iter_flatten( x )
  print( list(flat) )
  flat = np.flatten( x )
  print( flat )
  
  xhat = np.pack_sequence_as( x, flat )
  print( xhat )
  
  print( np.zip_with( tuple, [x, y, z] ) )
  
if __name__ == "__main__":
  test()
