# l2m_data

Dedicated LFS repo for large data files related to the SRI L2M SG.

If you're new to LFS, here is a decent starting point: https://docs.gitlab.com/ee/topics/git/lfs/

## Recommended `git` workflow

To fine-tune what files get pulled when you do `git pull`, use `git config lfs.fetchinclude XXX` and/or `git config lfs.fetchexclude YYY`

**Example**: To clone an initially-empty repository and then add files incrementally:

* Clone the repo, excluding all LFS files:
```bash
$ git clone -c lfs.fetchexclude='*' https://gitlab.sri.com/l2m/system/l2m_data.git
```

* Pull only the data files you want:
```bash
$ cd l2m_data
$ git config lfs.fetchexclude ''
$ git config lfs.fetchinclude 'task_labels.npy,training_data.npz'
$ git lfs pull
```

# `trajectories/` datasets

These are datasets of complete trajectories generated from a Task+Policy pair. The directory structure will be `trajectories/<name of policy>/<name of task>/epXXX.pkl.gz`, where `XXX` is the sequential index of the episode.

The trajectories can be loaded with `m3t.data.load_trajectory( <path> )`. The implementation of this function in `l2m_data` is identical in functionality and compatible with the version in `l2m_full`, but without dependencies on other parts of `l2m_full`.

## Data format

Each trajectory is a `namedtuple` of type `m3t.data.Trajectory`. It has fields `observations`, `actions`, and `rewards`. The `observations` and `actions` fields are `tuple`s of `np.ndarray`s, while `rewards` is a single `np.ndarray`. All of the `numpy` arrays have time in axis `0`.

* `observations` is a tuple `(feature_screen, available_actions, rgb_screen)`, where:
  * `feature_screen` has shape `(T, C=27, H=64, W=64)`. These are "image-like" structures where each channel corresponds to a semantic feature. More details below.
  * `available_actions` has shape `(T, A=13)`. Each timestep is a Boolean vector indicating whether each of the 13 actions are legal to execute in the corresponding state. More details on actions below.
  * `rgb_screen` has shape `(T, H=256, W=256, C=3)`. It is a "natural image" observation of the game similar to what a human player would see (but only the main view area, not the UI or the minimap). It represents the same viewport as `feature_screen` but at higher resolution (but note that due to the camera perspective, it actually doesn't see the entire area; the camera is off to the side slightly). 
* `actions` is a list of length `6`. Each element is a `numpy` array of shape `(T,)` containing the values of one action factor. More details about actions below.
* `rewards` is a `numpy` array of shape `(T,)`

## Feature selection for `feature_screen`

The full list of semantic feature layers in `feature_screen` is as follows (c.f. `pysc2.lib.features.ScreenFeatures`):
```python
[
  "height_map",
  "visibility_map",
  "creep",
  "power",
  "player_id",
  "player_relative",
  "unit_type",
  "selected",
  "unit_hit_points",
  "unit_hit_points_ratio",
  "unit_energy",
  "unit_energy_ratio",
  "unit_shields",
  "unit_shields_ratio",
  "unit_density",
  "unit_density_aa",
  "effects",
  "hallucinations",
  "cloaked",
  "blip",
  "buffs",
  "buff_duration",
  "active",
  "build_progress",
  "pathable",
  "buildable",
  "placeholder"
]
```
With the above list (order is important!), `feature_names.index( <name> )` will give you the index of the corresponding channel in the `feature_screen` array. The same list can be obtained from `pysc2.lib.features.ScreenFeatures._fields`.

### Standardized feature sets

Most of the features in `feature_screen` are not actually useful for the limited tasks we consider, and many of them will always just be zeros. Therefore, we propose a few "named" subsets of the features. We ask that everyone start with the smallest feature set (the `minimal` set) and get their components working before proceeding (if desired) to the more-complex sets.

**When extracting the feature subset from the full `feature_screen` array, the order of the channels must be the same as the order in `pysc2.lib.features.ScreenFeatures._fields`.**

#### The `minimal` set: 
```python
["unit_type", "selected", "unit_density"]
```

We consider this the minimal set of features that allow for effective RL policy learning in the existing Starcraft tasks (except for the "build" tasks), and this is the set of features we have been using in our system. **Everyone's components should, at a minimum, work with this feature set.** (Unless your component consumes natural images instead).

#### The `combat` set: 
```python
[
  "visibility_map",
  "player_relative",
  "unit_type",
  "selected",
  "unit_hit_points_ratio",
  "unit_shields_ratio",
  "unit_density"
]
```

This feature set would, in principle, allow somewhat better optimal performance on the existing Starcraft tasks (but is still not sufficient for the "build" tasks). 

#### The `build` set: 
```python
[
  "height_map", 
  "visibility_map", 
  "creep",
  "power",
  "player_relative", 
  "unit_type", 
  "selected", 
  "unit_hit_points_ratio",
  "unit_energy",
  "unit_shields_ratio", 
  "unit_density",
  "build_progress",
  "pathable",
  "buildable"
]
```

This set extends the `combat` set with the features that are necessary for tasks that involve constructing buildings.

## Actions

Starcraft (via `pysc2`) uses a *factored action space* (c.f. `pysc2.lib.actions`). Conceptually, an action is represented as `function(*args)` (i.e., a function and 0 or more arguments). The full action space is **huge**. To make the problem simpler, we selected a minimal set of action functions, and we only consider the argument types that are relevant to the selected functions.

The action set in the `trajectories` datasets is:
```python
[
  0,   # no_op
  1,   # move_camera
  2,   # select_point
  3,   # select_rect
  12,  # Attack_screen
  140, # Cancel_quick
  168, # Cancel_Last_quick
  261, # Halt_quick
  274, # HoldPosition_quick
  331, # Move_screen
  333, # Patrol_screen
  451, # Smart_screen
  453, # Stop_quick
]
```

In the `actions` field of a trajectory, `actions[0]` contains indices into this list. The rest of `actions[1:]` contains arguments, in the following order:
```python
[
  "screen",
  "minimap",
  "screen2",
  "select_add",
  "select_point_act"
]
```

**Note: The order of arguments in `actions` is different from the order in `pysc2.lib.actions.Arguments`. This is due to an oversight, but we're stuck with it.**

* The arguments `screen`, `minimap`, and `screen2` are positions in the coordinate system of `feature_screen` represented as an integer `i`, such that `y, x = divmod(i, 64)`. 
* `select_add` is a Boolean feature specifying whether a "select" action should replace the current selection, or add to it
* `select_point_act` has 4 possible values that cause a point-select action to have different effects

The neural network policies always output values for all arguments. The arguments that don't apply to the action function (`actions[0]`) get masked out. 

The `available_actions` field in `observations` specifies which of the action functions are available in the current state. The policy network is forbidden from selecting invalid actions by masking out the invalid action functions and then sampling from what's left.


# Task syllabi (a/k/a curricula)

We have been using the following two task orders for our full-system experiments (`dm5` = this set of 5 DeepMind tasks):

* `dm5_A`: `CollectMineralShards, DefeatRoaches, MoveToBeacon, DefeatZerglingsAndBanelings, FindAndDefeatZerglings`
* `dm5_B`: `CollectMineralShards, MoveToBeacon, FindAndDefeatZerglings, DefeatRoaches, DefeatZerglingsAndBanelings`

Note that in syllabus `A`, similar tasks (e.g., `DefeatRoaches, DefeatZerglingsAndBanelings`) are mixed up, while in syllabus `B`, similar tasks are grouped together. In our full runs we repeat this order 6 times (but we also use see a lot more than 100 total episodes of each).


# Example network architecture

Our policy networks take the `minimal` subset of `feature_screen` as input, and they produce the necessary outputs for actor-critic training -- a value estimate, and the logits for each of the action factors discussed above.

Here is a summary of our architecture, which is based on the `FullyConv` model from [Vinyals et al., 2017]:

1. Feature encoding
	* Categorical features (those with `FeatureType.CATEGORICAL` in `pysc2.lib.features.SCREEN_FEATURES`) encoded with `tf.keras.layers.Embedding` with `output_dim=int(max(1, round(np.log2(feature.scale))))`, then transposed back to channels-first order
	* Scalar features are log-transformed
2. Conv layers
	1. `Conv2D(channels=16, filter_size=5, padding="same", activation="relu")`
	2. `Conv2D(channels=32, filter_size=3, padding="same", activation="relu")`
3. Spatial dimension reduction
	* This is done because in the `FullyConv` architecture, `(x, y)` locations in the 2D features correspond to the possible `(x, y)` arguments to actions, and we want to reduce the action space
	* `Conv2D(channels=16, filter_size=(2*reduction_factor - 1), strides=reduction_factor, padding="same", activation="relu")`
	* By default we reduce `64x64` to `16x16`, so `reduction_factor=4`
4. Outputs
	* Create `fc` features: `Flatten()` -> `Dense(256, activation="relu")`
	* Value output (for actor-critic): `fc` -> `Dense(1)`
	* For each action factor, output a vector of logits:
		* Non-spatial factor: `fc` -> `Dense(num_possible_values)`
		* Spatial factors: `Conv2D(channels=1, filter_size=1)` -> `Flatten()`
			* We will later "un-pack" the spatial argument as `y, x = divmod(output, 16)` and then scale `x` and `y` with `int((x+0.5) * reduction_factor)` to get back to the `64x64` spatial dimension.


# Legacy datasets

The following information is preserved for posterity, but you should use one of the better datasets described above.

## `training_data.npz` and `task_labels.npy`

This is a numpy zip archive containing agent observations from 68718 timesteps of execution of a random policy in 5 different Starcraft mini-games. They are labeled with the name of the mini-game from which the observation was generated.

### Loading

```python
import numpy as np
...
# Use context manager to avoid leaking file descriptor
with np.load("training_data.npz", allow_pickle=True) as data:
  # Labels don't need to be closed because they aren't zipped
  labels = np.load("task_labels.npy", allow_pickle=True)
  # Data must be accessed by dict keys
  for (i, k) in enumerate(data.keys()):
    obs = data[k]
    label_string = labels[i]
```

### Observation format

Each entry in the dataset is a numpy object array with 18 elements. Indices `0...16` are each `64x64` image-like layers, each one corresponding to a different semantic feature. Index `17` gives the reward received in that timestep.

These data were collected with `pysc2` version `2.0.2`. **We use `pysc2` version `3` now.** Version 3 has additional feature layers that are a superset of the Version 2 features. The Version 2 features in this dataset are as follows:
```python
feature_names = [
  "height_map",
  "visibility_map",
  "creep",
  "power",
  "player_id",
  "player_relative",
  "unit_type",
  "selected",
  "unit_hit_points",
  "unit_hit_points_ratio",
  "unit_energy",
  "unit_energy_ratio",
  "unit_shields",
  "unit_shields_ratio",
  "unit_density",
  "unit_density_aa",
  "effects"
]
```
With the above list (order is important!), `feature_names.index("unit_type")` will give you the index of the `unit_type` layer in the observation array.

### Feature Selection and Preprocessing

The most important feature is `unit_type`. Our policies currently use the input features `["unit_type", "selected", "unit_density"]`, but that may change in the future. If you want to use the same representation we use for our policies, you can use the following code (you will need the `l2m_full` package installed):
```python
import reaver.models.sc2.vae
...
# Do this once at start-up:
reaver.models.sc2.vae.configure_observation_normalizer( 
  screen_features=["unit_type", "selected", "unit_density"] )
...
# When loading data:
for k in data.keys():
  raw_obs = data[k]
  # pysc2 data always uses channels-first order
  raw_obs = np.stack( raw_obs[:-1], axis=0 )
  # Set batch_dims > 0 if you have leading batch dimension(s)
  obs = reaver.models.sc2.vae.normalize_observations( raw_obs, batch_dims=0 )
```

### Label Format

The `i`th entry in `task_labels` is the string name of the task from which the corresponding observation was generated. To obtain a canonical integer encoding of the label, index into a **sorted** list of task names:
```python
label_strings = np.load("task_labels.npy", allow_pickle=True)
task_names = sorted(set(label_strings))
label_integers = np.array( [task_names.index(s) for s in label_strings] )
```

# License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

© SRI International 2022. This work is licensed under
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR0011-18-C-0051.

Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Defense Advanced Research Projects Agency (DARPA).