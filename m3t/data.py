import collections
import gzip
import pathlib
import pickle

import numpy as np

import m3t.nest


Trajectory = collections.namedtuple( 
  "Trajectory", ["observations", "actions", "rewards"] )
Trajectory.__doc__ = """
The data from a single trajectory of a policy in a Starcraft task.

Each field is either a numpy array or a tuple of numpy arrays. The numpy
arrays have shape[0] equal to the number of time steps.

Fields:
  observations: A tuple (feature_screen, available_actions, rgb_screen)
  actions: A tuple of action factors, where the first element is the action
    "function" and the rest are arguments.
  rewards: A numpy array containing per-step rewards.
"""
  

def load_trajectory( path, color_order="BGR", gzipped=None ):
  """ Loads a trajectory, returning a `Trajectory` instance.
  
  Parameters:
    path: The path to the trajectory file
    color_order: The order of the channels in the natural-image observations
      (these observations are called 'rgb_screen' in pysc2, but for technical
      reasons the data that SRI provides will usually have the colors in
      "BGR" order).
    gzipped: Specify True or False as appropriate if the trajectory file is
      gzipped. If not specified, inferred from the file extension (.gz means
      a gzip file).
      
  Returns:
    A Trajectory instance
  """
  if color_order not in ["BGR", "RGB"]:
    raise ValueError( "color_order must be in ['BGR', 'RGB']" )

  if gzipped is None:
    ext = pathlib.Path(path).suffix
    gzipped = (ext == ".gz")
  
  def open_trajectory_file( f ):
    if gzipped:
      return gzip.open( f, "rb" )
    else:
      return open( f, "rb" )
  
  with open_trajectory_file( path ) as file:
    traj = pickle.load( file )
    if color_order == "BGR":
      img = traj.observations[2]
      # Reverse the last axis (channels)
      img = img[:,:,:,::-1]
      obs = (traj.observations[0], traj.observations[1], img)
      traj = traj._replace(observations=obs)
    return traj
