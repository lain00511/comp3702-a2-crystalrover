import sys

from game_env import GameEnv
from gui import Viewer

"""
play_game.py

Running this file launches an interactive game session. Becoming familiar with the game mechanics may be helpful in
designing your solution.

The script takes 1 argument:
- input_filename, which must be a valid testcase file (e.g. one of the provided files in the testcases directory)

When prompted for an action, type one of the available action strings (e.g. wr, wl, etc) and press enter to perform the
entered action.

COMP3702 Assignment 2 "CrystalRover" Support Code

Last updated by vp 09/09/2026
"""


def main(arglist):
    if len(arglist) not in {1, 2}:
        print("Running this file launches an interactive game session.")
        print("Usage: play_game.py [input_filename] [--no-storm-particles]")
        return -1

    input_file = arglist[0]
    enable_storm_particles = True
    if len(arglist) == 2:
        if arglist[1] == '--no-storm-particles':
            enable_storm_particles = False
        else:
            print("Unknown option:", arglist[1])
            print("Usage: play_game.py [input_filename] [--no-storm-particles]")
            return -1

    game_env = GameEnv(input_file)
    gui = Viewer(game_env, enable_storm_particles=enable_storm_particles)
    persistent_state = game_env.get_init_state()
    actions = []
    total_cost = 0

    print(f'Loaded level from {input_file}.')
    print(f'Storm info : {game_env.storm_direction} at {game_env.storm_directional_cost}')

    # run simulation
    while True:
        gui.update_state(persistent_state)
        valid_actions = game_env.ACTIONS
        print(f"Total cost so far: {round(total_cost, 1)}")
        print(f'Actions: {valid_actions} | Enter "q" to quit.')
        print('Choose an action >>', end=' ')
        a = input().strip()
        if 'q' in a:
            print('Quitting.')
            break
        if a not in valid_actions:
            print('Invalid action. Choose again.')
            continue
        actions.append(a)
        persistent_state, action_cost, err_msg = game_env.perform_action(persistent_state, a)
        total_cost += action_cost
        if err_msg is not None:
            print(f'{err_msg}')
        if game_env.is_solved(persistent_state):
            gui.update_state(persistent_state)
            print(f'Level completed with total cost of {round(total_cost, 1)}!')
            break
        elif game_env.is_game_over(persistent_state):
            gui.update_state(persistent_state)
            print(f'Game Over. total cost = {round(total_cost, 1)}')
            break

    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
