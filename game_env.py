from game_state import GameState
import random

"""
game_env.py

COMP3702 Assignment 2 "CrystalRover" Support Code

Last updated by vp 09/09/2026
"""


class GameEnv:
    """
    Instance of an Crystal Rover Game environment. Stores the dimensions of the environment, initial rover position,
    crater positions, rock positions, storm direction and cost, exit positions, crystal sample positions
    and the available actions.
    """

    # input file symbols
    GROUND_TILE = ' '
    ROCK_TILE = 'R'
    CRATER_TILE = '*'
    CRYSTAL_TILE = 'C'
    LAUNCH_TILE = 'E'
    ROVER_TILE = 'P'
    LAVA_TILE = 'L'
    VALID_TILES = {GROUND_TILE, ROCK_TILE, CRATER_TILE, CRYSTAL_TILE, LAUNCH_TILE, ROVER_TILE, LAVA_TILE}

    # action symbols (i.e. output file symbols)
    WALK_LEFT = 'wl'
    WALK_RIGHT = 'wr'
    WALK_UP = 'wu'
    WALK_DOWN = 'wd'
    BOOST_LEFT = 'bl'
    BOOST_RIGHT = 'br'
    BOOST_UP = 'bu'
    BOOST_DOWN = 'bd'
    JUMP_LEFT = 'jl'
    JUMP_RIGHT = 'jr'
    JUMP_UP = 'ju'
    JUMP_DOWN = 'jd'

    ACTIONS = [
        WALK_LEFT, WALK_RIGHT, WALK_UP, WALK_DOWN,
        BOOST_LEFT, BOOST_RIGHT, BOOST_UP, BOOST_DOWN,
        JUMP_LEFT, JUMP_RIGHT, JUMP_UP, JUMP_DOWN,
    ]
    ACTION_BASE_COST = {
        WALK_LEFT: 1.0, WALK_RIGHT: 1.0, WALK_UP: 1.0, WALK_DOWN: 1.0,
        BOOST_LEFT: 1.5, BOOST_RIGHT: 1.5, BOOST_UP: 1.5, BOOST_DOWN: 1.5,
        JUMP_LEFT: 2.0, JUMP_RIGHT: 2.0, JUMP_UP: 2.0, JUMP_DOWN: 2.0,
    }
    WALK_ACTIONS = {WALK_LEFT, WALK_RIGHT, WALK_UP, WALK_DOWN}
    BOOST_ACTIONS = {BOOST_LEFT, BOOST_RIGHT, BOOST_UP, BOOST_DOWN}
    JUMP_ACTIONS = {JUMP_LEFT, JUMP_RIGHT, JUMP_UP, JUMP_DOWN}


    PERPENDICULAR_ACTIONS = {
        WALK_LEFT: [WALK_UP, WALK_DOWN],
        WALK_RIGHT: [WALK_UP, WALK_DOWN],
        WALK_UP: [WALK_LEFT, WALK_RIGHT],
        WALK_DOWN: [WALK_LEFT, WALK_RIGHT],
        BOOST_LEFT: [BOOST_UP, BOOST_DOWN],
        BOOST_RIGHT: [BOOST_UP, BOOST_DOWN],
        BOOST_UP: [BOOST_LEFT, BOOST_RIGHT],
        BOOST_DOWN: [BOOST_LEFT, BOOST_RIGHT],
        JUMP_LEFT: [JUMP_UP, JUMP_DOWN],
        JUMP_RIGHT: [JUMP_UP, JUMP_DOWN],
        JUMP_UP: [JUMP_LEFT, JUMP_RIGHT],
        JUMP_DOWN: [JUMP_LEFT, JUMP_RIGHT],
    }

    # perform action return statuses
    SUCCESS = 0
    GAME_OVER = 2

    def __init__(self, filename):
        """
        Parse the supplied testcase file and initialise the environment.
        """
        try:
            with open(filename, 'r') as f:
                lines = [line.rstrip('\n') for line in f]
        except FileNotFoundError:
            assert False, '/!\\ ERROR: Testcase file not found'

        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            cleaned.append(stripped)

        assert len(cleaned) >= 9, '/!\\ ERROR: Invalid input file - expected header and grid data'

        try:
            self.n_rows, self.n_cols = tuple(int(x) for x in cleaned[0].split(','))
            self.num_rocket_jumps = cleaned[1] # infinite now :)
            self.min_samples = int(cleaned[2])
            self.storm_direction = cleaned[3].upper()
            self.storm_directional_cost = float(cleaned[4])
            self.gamma, self.epsilon = tuple(float(x) for x in cleaned[5].split(','))
            self.vi_time_min_tgt, self.vi_time_max_tgt = tuple([float(x) for x in cleaned[6].split(',')])
            self.pi_time_min_tgt, self.pi_time_max_tgt = tuple([float(x) for x in cleaned[7].split(',')])
            self.vi_iter_min_tgt, self.vi_iter_max_tgt = tuple([int(x) for x in cleaned[8].split(',')])
            self.pi_iter_min_tgt, self.pi_iter_max_tgt = tuple([int(x) for x in cleaned[9].split(',')])
            self.reward_min_tgt, self.reward_max_tgt = tuple([float(x) for x in cleaned[10].split(',')])
            self.random_drift_prob = float(cleaned[11])
            self.random_double_prob = float(cleaned[12])
            self.boost_probabilities = tuple([float(x) for x in cleaned[13].split(',')])
            self.collision_penalty = float(cleaned[14])
            self.game_over_penalty = float(cleaned[15])
            self.episode_seed = int(cleaned[16])
            
        except (ValueError, IndexError):
            assert False, '/!\\ ERROR: Invalid input file - malformed header values'

        random.seed(self.episode_seed)
        grid_lines = cleaned[17:17 + self.n_rows]
        assert len(grid_lines) == self.n_rows, '/!\\ ERROR: Invalid input file - incorrect number of map rows - expected {} but got {}'.format(self.n_rows, len(grid_lines))

        self.grid_data = []
        self.init_row, self.init_col = None, None
        self.crystal_positions = []
        self.launch_positions = [] 
        self.lava_positions = []

        for r, row in enumerate(grid_lines):
            chars = list(row)
            assert len(chars) == self.n_cols, '/!\\ ERROR: Invalid input file - incorrect map row length - expected {} but got {}'.format(self.n_cols, len(chars))
            for c, ch in enumerate(chars):
                if ch == self.ROVER_TILE:
                    assert self.init_row is None and self.init_col is None, '/!\\ ERROR: Invalid input file - more than one initial player position'
                    self.init_row, self.init_col = r, c
                    chars[c] = self.GROUND_TILE
                elif ch == self.LAUNCH_TILE:
                    self.launch_positions.append((r, c))
                    chars[c] = self.LAUNCH_TILE
                elif ch == self.CRYSTAL_TILE:
                    self.crystal_positions.append((r, c))
                    chars[c] = self.GROUND_TILE
                elif ch == self.LAVA_TILE:
                    self.lava_positions.append((r, c))
                    chars[c] = self.LAVA_TILE
            self.grid_data.append(chars)

        assert self.init_row is not None and self.init_col is not None, '/!\\ ERROR: Invalid input file - No player initial position'
        assert len(self.launch_positions) > 0, '/!\\ ERROR: Invalid input file - No launch position'

        self.n_crystals = len(self.crystal_positions)
        self.all_crystals_tuple = tuple([1 for _ in range(self.n_crystals)])
        self.ACTION_COST = self._build_action_costs()

    def _build_action_costs(self):
        action_costs = {}
        for action, base_cost in self.ACTION_BASE_COST.items():
            direction = self._action_direction(action)
            if direction is None:
                cost = base_cost
            elif direction == self.storm_direction:
                cost = base_cost - self.storm_directional_cost
            elif direction == self._opposite_direction(self.storm_direction):
                cost = base_cost + self.storm_directional_cost
            else:
                cost = base_cost
            action_costs[action] = cost
        return action_costs

    def _action_direction(self, action):
        mapping = {
            self.WALK_LEFT: 'LEFT', self.BOOST_LEFT: 'LEFT', self.JUMP_LEFT: 'LEFT',
            self.WALK_RIGHT: 'RIGHT', self.BOOST_RIGHT: 'RIGHT', self.JUMP_RIGHT: 'RIGHT',
            self.WALK_UP: 'UP', self.BOOST_UP: 'UP', self.JUMP_UP: 'UP',
            self.WALK_DOWN: 'DOWN', self.BOOST_DOWN: 'DOWN', self.JUMP_DOWN: 'DOWN',
        }
        return mapping.get(action)

    def _opposite_direction(self, direction):
        return {
            'LEFT': 'RIGHT',
            'RIGHT': 'LEFT',
            'UP': 'DOWN',
            'DOWN': 'UP',
        }.get(direction)


    def get_init_state(self):
        """
        Get a state representation instance for the initial state.
        :return: initial state
        """
        state = GameState(self.init_row, self.init_col, tuple(0 for _ in self.crystal_positions))
        return state

    def perform_action(self, state, action):
        """
        Perform the given action on the given state and return whether the action was successful
        plus the resulting new state and reward.
        """

        if action not in self.ACTIONS:
            return state, 0.0, "Invalid action"

        # apply drift and double probs
        movements = self.apply_action_noise(action)

        # apply dynamics based on movements
        new_state = state.deepcopy()
        total_reward = 0.0
        err_msg = None
        for m in movements:
            valid, err_msg, next_state, reward, terminal_state = self.apply_dynamics(new_state, m)
            if not valid:
                continue
            new_state = next_state
            total_reward += reward
            if terminal_state:
                break
        return new_state, total_reward, err_msg


    def is_solved(self, state):
        """
        Check if the game has been solved (i.e. player at exit and minimum number of crystals collected)
        :param state: current GameState
        :return: True if solved, False otherwise
        """
        collected = sum(state.crystal_status)
        return (state.row, state.col) in self.launch_positions and collected >= self.min_samples

    def is_game_over(self, state):
        """
        The game is over if the player is in lava.
        :param state: current GameState
        :return: True if game is over, False otherwise
        """
        return (self.grid_data[state.row][state.col] == self.LAVA_TILE)

    def render(self, state):
        """
        Render the map's current state to terminal.
        """
        for r in range(self.n_rows):
            line = ''
            for c in range(self.n_cols):
                if state.row == r and state.col == c:
                    line += 'P'
                elif (r, c) in self.launch_positions:
                    line += 'E'
                elif (r, c) in self.lava_positions:
                    line += 'L'
                elif (r, c) in self.crystal_positions and state.crystal_status[self.crystal_positions.index((r, c))] == 0:
                    line += 'C'
                else:
                    line += self.grid_data[r][c]
            print(line)
        print('\n' * 2)
    

    def apply_action_noise(self, action):
        """
        Apply the action noise to the given action and return the resulting action.
        :param action: action string
        :return: resulting action string
        """
        movements = []

        drifted = random.random() < self.random_drift_prob
        doubled = random.random() < self.random_double_prob

        if drifted:
            # randomly drift perpendicular to the intended direction
            perpendiculars = self.PERPENDICULAR_ACTIONS[action]
            drift_choice = random.choice(perpendiculars)
            movement = drift_choice
        else:
            movement = action

        movements.append(movement)
        if doubled:
            movements.append(movement)

        return movements

    def apply_dynamics(self, state, action):
        """
        Apply the dynamics of the game to the given state and action and return the resulting state and reward.
        :param state: current GameState
        :param action: action string
        :return: action is valid (True/False), error message if invalid, next state, reward, state is terminal
        """

        if action not in self.ACTIONS:
            return False, "Invalid action", None, 0.0, None

        reward = -1 * self.ACTION_COST[action]
        next_row, next_col = state.row, state.col

        direction = self._action_direction(action)

        deltas = {
            'LEFT': (0, -1),
            'RIGHT': (0, 1),
            'UP': (-1, 0),
            'DOWN': (1, 0),
        }

        delta_row, delta_col = deltas[direction]

        if action in self.JUMP_ACTIONS:
            if self.grid_data[state.row][state.col] != self.CRATER_TILE:
                return False, "Cannot perform rocket jump", None, 0.0, None
            move_distance = 1
        elif action in self.WALK_ACTIONS:
            if self.grid_data[state.row][state.col] == self.CRATER_TILE:
                return False, "Cannot perform action: in a crater", None, 0.0, None
            move_distance = 1
        elif action in self.BOOST_ACTIONS:
            if self.grid_data[state.row][state.col] == self.CRATER_TILE:
                return False, "Cannot perform action: in a crater", None, 0.0, None
            # sample the boost distance based on the boost probabilities
            move_distance = random.choices([0, 1, 2, 3, 4], weights=self.boost_probabilities, k=1)[0]

        collision = False
        for _ in range(move_distance):
            candidate_row = next_row + delta_row
            candidate_col = next_col + delta_col
            if not (0 <= candidate_row < self.n_rows and 0 <= candidate_col < self.n_cols) \
                    or self.grid_data[candidate_row][candidate_col] == self.ROCK_TILE:
                reward -= self.collision_penalty
                collision = True
                break

            next_row, next_col = candidate_row, candidate_col

            # fall into a crater
            if self.grid_data[next_row][next_col] == self.CRATER_TILE:
                break

            # fall into lava
            if self.grid_data[next_row][next_col] == self.LAVA_TILE:
                reward -= self.game_over_penalty
                break

        crystal_status = state.crystal_status
        if (next_row, next_col) in self.crystal_positions:
            crystal_index = self.crystal_positions.index((next_row, next_col))
            if crystal_status[crystal_index] == 0:
                crystal_status = list(crystal_status)
                crystal_status[crystal_index] = 1
                crystal_status = tuple(crystal_status)

        next_state = GameState(next_row, next_col, crystal_status)
        if not collision and self.is_game_over(next_state) and \
                self.grid_data[next_row][next_col] != self.LAVA_TILE:
            reward -= self.game_over_penalty

        return True, None, next_state, reward, self.is_game_over(next_state)