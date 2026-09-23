# COMP3702 Assignment 2 Support Code

This is the support code for COMP3702 2026 Assignment 2, based on the CrystalRover game.

## Files

### game_env.py

This file defines the `GameEnv` class and contains the environment dynamics used by the assignment.

It stores the following key data:
- `n_rows`, `n_cols`
- `init_row`, `init_col`
- `crystal_positions`
- `launch_positions`
- `lava_positions`
- `storm_direction`, `storm_directional_cost`
- `gamma`, `epsilon`
- `random_drift_prob`, `random_double_prob`
- `boost_probabilities`, `collision_penalty`, `game_over_penalty`
- `ACTIONS`, `ACTION_BASE_COST`

Important methods:

~~~~~
__init__(filename)
~~~~~
Parses the testcase file and initialises the environment.

~~~~~
get_init_state()
~~~~~
Returns a GameState object (see below) representing the initial state of the level.

~~~~~
perform_action(state, action)
~~~~~
Applies the action to a state and returns the resulting state and reward. The action is stochastic: random drift and double-action noise may modify the intended move.

~~~~~
is_solved(state)
~~~~~
Returns `True` when the rover is on a launch tile and has collected at least `min_samples` crystals.

~~~~~
is_game_over(state)
~~~~~
Returns `True` if the rover is on a lava tile.

~~~~~
render(state)
~~~~~
Prints the current environment state to the terminal.

~~~~~
apply_action_noise(action)
~~~~~
Applies stochastic drift and double-step effects to the action.

~~~~~
apply_dynamics(state, action)
~~~~~
Applies the deterministic movement rules for the action and returns the next state, reward, and terminal flag.

### game_state.py

This file defines the `GameState` object used throughout the solver and tester.

A `GameState` stores:
- `row`
- `col`
- `crystal_status` as a tuple of 0/1 values, where 1 means that crystal has been collected and 0 means it remains.

Key methods:

~~~~~
__init__(row, col, crystal_status)
~~~~~
Creates a game state with the rover's position and crystal collection status.

~~~~~
__eq__(other)
~~~~~
Compares two states.

~~~~~
__hash__()
~~~~~
Allows use as a dictionary key.

~~~~~
deepcopy()
~~~~~
Returns a copy of the state.

### play_game.py

This script launches an interactive CrystalRover session.

Usage:

```bash
python play_game.py [input_filename] [--no-storm-particles]
```

The script loads a testcase, displays the board, and prompts the user for an action. Valid actions are the entries in `GameEnv.ACTIONS`.

### gui.py

This file contains the visualiser used by the game and by the interactive play session. It renders the world, the crystals, the rover, and storm effects in a graphical window.

### solution.py

This is the template in which you implement your solver.

You must implement the following method stubs, which will be invoked by the simulator during testing:

- `__init__(game_env)`
- `plan_offline()`
- `select_action()`

Implementing a 'main' method is not required.

To ensure compatibility with the autograder, please avoid using try-except blocks for Exception or OSError exception
types. Try-except blocks with concrete exception types other than OSError (e.g. try: ... except ValueError) are allowed.

We recommend you start by implementing value iteration (including developing a transition model), and work to improve
your implementation from here.

### tester.py

This script is used to debug and evaluate a candidate solver.

Usage:

```bash
python tester.py [vi|pi] [testcase_file] [-v]
```

### testcases/

This directory contains example input files for the CrystalRover assignment.
