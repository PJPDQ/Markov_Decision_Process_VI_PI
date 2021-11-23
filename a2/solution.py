import math
import time
import random
import numpy as np
import itertools

from game_env import GameEnv
from game_state import GameState

"""
solution.py

Template file for you to implement your solution to Assignment 2.

You must implement the following method stubs, which will be invoked by the simulator during testing:
    __init__(game_env)
    plan_offline()
    select_action()

To ensure compatibility with the autograder, please avoid using try-except blocks for Exception or OSError exception
types. Try-except blocks with concrete exception types other than OSError (e.g. try: ... except ValueError) are allowed.

COMP3702 2021 Assignment 2 Support Code

Last updated by njc 02/09/21
"""


class Solver:

    def __init__(self, game_env):
        """
        Constructor for your solver class.

        Any additional instance variables you require can be initialised here.

        Computationally expensive operations should not be included in the constructor, and should be placed in the
        plan_offline() method instead.

        This method has an allowed run time of 1 second, and will be terminated by the simulator if not completed within
        the limit.
        """
        self.game_env = game_env
        self.solver_type = 'vi'
        self.reachable_actions = {}  # key: action, value: (row, col)
        self.reach_states = {}  # key: reachable_state, value: reward
        self.reachable_states = []
        self.reachable_pos = set()
        self.policy = {}
        self.values = {}

        ## Value Iteration and Policy Iteration
        self._actions = [self.game_env.WALK_LEFT, self.game_env.WALK_RIGHT, self.game_env.JUMP,
                         self.game_env.GLIDE_LEFT_1, self.game_env.GLIDE_LEFT_2, self.game_env.GLIDE_LEFT_3,
                         self.game_env.GLIDE_RIGHT_1, self.game_env.GLIDE_RIGHT_2, self.game_env.GLIDE_RIGHT_3,
                         self.game_env.DROP_1, self.game_env.DROP_2, self.game_env.DROP_3]
        self.gamma = 0.9999
        self.epsilon = 0.001
        self.init_state = self.game_env.get_init_state()
        self.rewards = {}  # REWARDS.keys() PI
        for a in self.game_env.ACTIONS:
            is_valid, reward, next_state, _ = self.game_env.perform_action(self.init_state, a)
            if is_valid:
                # self.states.append(next_state)
                # self.actions.append(a)
                self.rewards[(next_state.row, next_state.col)] = reward
                self.policy[a] = next_state
                self.values[next_state] = reward
        self.converged = False
        self.USE_LIN_ALG = True
        self.EXIT_STATE = GameState(self.game_env.exit_row, self.game_env.exit_col,
                                    tuple(1 for g in self.game_env.gem_positions))
        # self.t_model = np.zeros([len(self.states), len(self.actions), len(self.states)])
        # self.r_model = np.zeros([len(self.states)])
        # self.la_policy = np.zeros([len(self.states)])
        # self.res_policy = {}

    def get_reward(self, state):
        if (state.row, state.col) == self.EXIT_STATE:
            return 0
        if (state.row, state.col) in self.rewards:
            return self.rewards[(state.row, state.col)]
        else:
            return 0

    def get_reachable_action(self, curr_state):
        actions = []
        # pos = self.game_env.gem_positions[0]
        pos = (curr_state.row, curr_state.col)
        for action in self.game_env.ACTIONS:
            if (action in {self.game_env.WALK_LEFT, self.game_env.WALK_RIGHT, self.game_env.JUMP}
                and self.game_env.grid_data[pos[0] + 1][pos[1]] not in self.game_env.WALK_JUMP_ALLOWED_TILES) \
                    or ((action in self.game_env.GLIDE_ACTIONS or action in self.game_env.DROP_ACTIONS)
                        and self.game_env.grid_data[pos[0] + 1][pos[1]] not in self.game_env.GLIDE_DROP_ALLOWED_TILES):
                pass
            else:
                actions.append(action)
        return actions

    def get_reachable_position_and_actions(self):
        # because the gem is always reachable, so best to start from the gem position
        init_pos = self.game_env.gem_positions[0]
        # init_pos = (self.game_env.init_row, self.game_env.init_col)
        visited = {init_pos}
        container = [init_pos]
        valid_actions = {}  # key: (row, col), value: action
        terminal_states = {} # key: (row, col), action value: is_terminal
        rewards = {}

        while len(container) > 0:
            pos = container.pop(-1)
            curr_state = GameState(pos[0], pos[1], tuple(0 for g in self.game_env.gem_positions))
            for a in self.get_reachable_action(curr_state):
                is_valid, reward, next_state, is_terminal = self.game_env.perform_action(curr_state, a)
                if is_valid:
                    new_pos = (next_state.row, next_state.col)
                    terminal_states[(*pos, a)] = is_terminal
                    rewards[next_state] = reward
                    if pos in valid_actions.keys():
                        valid_actions[pos].append(a)
                    else:
                        valid_actions[pos] = [a]
                    if new_pos not in visited:
                        container.append(new_pos)
                        visited.add(new_pos)
        self.rewards = rewards
        return visited, valid_actions

    def get_reachable_status(self):
        return list(itertools.product(*[(0, 1)] * len(self.game_env.gem_positions)))

    def valid_actions(self, state, action):
        possible_actions = {}
        for move in self.game_env.ACTIONS:
            is_valid, reward, next_state, _ = self.game_env.perform_action(state, action)
            if is_valid:
                possible_actions[next_state] = reward
        return possible_actions

    def get_transition_probability(self, state, action):
        reward = -1 * GameEnv.ACTION_COST[action]
        remaining_prob = 1.0
        res = False, 0, 0, 0

        max_glide1_outcome = max(self.game_env.glide1_probs.keys())
        max_glide2_outcome = max(self.game_env.glide2_probs.keys())
        max_glide3_outcome = max(self.game_env.glide3_probs.keys())

        # handle each action type separately
        if action in GameEnv.WALK_ACTIONS:
            # set movement direction
            if action == GameEnv.WALK_LEFT:
                move_dir = -1
            else:
                move_dir = 1

            # walk on normal walkable tile (super charge case not handled)
            # if on ladder, handle fall case
            if self.game_env.grid_data[state.row + 1][state.col] == GameEnv.LADDER_TILE and \
                    self.game_env.grid_data[state.row + 2][state.col] not in GameEnv.COLLISION_TILES:
                next_row, next_col = state.row + 2, state.col
                # check if a gem is collected or goal is reached
                next_gem_status, _ = self.check_gem_collected_or_goal_reached(self.game_env, next_row, next_col,
                                                                              state.gem_status)
                # res = GameState(next_row, next_col, next_gem_status), reward, self.game_env.ladder_fall_prob
                remaining_prob -= self.game_env.ladder_fall_prob

            next_row, next_col = state.row, state.col + move_dir
            # check for collision or game over
            next_row, next_col, collision, is_terminal = \
                self.check_collision_or_terminal(self.game_env, next_row, next_col, row_move_dir=0,
                                                 col_move_dir=move_dir)

            # check if a gem is collected or goal is reached
            next_gem_status, _ = self.check_gem_collected_or_goal_reached(self.game_env, next_row, next_col,
                                                                          state.gem_status)

            if collision:
                # add any remaining probability to current state
                res = True, GameState(next_row, next_col, next_gem_status), \
                      reward - self.game_env.collision_penalty, remaining_prob
            elif is_terminal:
                # add any remaining probability to current state
                res = True, GameState(next_row, next_col, next_gem_status), \
                      reward - self.game_env.game_over_penalty, remaining_prob
            else:
                res = True, GameState(next_row, next_col, next_gem_status), reward, remaining_prob
        elif action == GameEnv.JUMP:
            # jump on normal walkable tile (super jump case not handled)

            next_row, next_col = state.row - 1, state.col
            # check for collision or game over
            next_row, next_col, collision, is_terminal = \
                self.check_collision_or_terminal(self.game_env, next_row, next_col, row_move_dir=-1, col_move_dir=0)

            # check if a gem is collected or goal is reached
            next_gem_status, _ = self.check_gem_collected_or_goal_reached(self.game_env, next_row, next_col,
                                                                          state.gem_status)

            if collision:
                # add any remaining probability to current state
                res = False, GameState(next_row, next_col,
                                       next_gem_status), reward - self.game_env.collision_penalty, 1.0
            elif is_terminal:
                # add any remaining probability to current state
                res = False, GameState(next_row, next_col,
                                       next_gem_status), reward - self.game_env.game_over_penalty, 1.0
            else:
                res = True, GameState(next_row, next_col, next_gem_status), reward, 1.0

        elif action in GameEnv.GLIDE_ACTIONS:
            # glide on any valid tile
            # select probabilities to sample move distance
            if action in {GameEnv.GLIDE_LEFT_1, GameEnv.GLIDE_RIGHT_1}:
                probs = self.game_env.glide1_probs
                max_outcome = max_glide1_outcome
            elif action in {GameEnv.GLIDE_LEFT_2, GameEnv.GLIDE_RIGHT_2}:
                probs = self.game_env.glide2_probs
                max_outcome = max_glide2_outcome
            else:
                probs = self.game_env.glide3_probs
                max_outcome = max_glide3_outcome

            # set movement direction
            if action in {GameEnv.GLIDE_LEFT_1, GameEnv.GLIDE_LEFT_2, GameEnv.GLIDE_LEFT_3}:
                move_dir = -1
            else:
                move_dir = 1

            # add each possible movement distance to set of outcomes
            next_row, next_col = state.row + 1, state.col
            for d in range(0, max_outcome + 1):
                next_col = state.col + (move_dir * d)
                # check for collision or game over
                next_row, next_col, collision, is_terminal = \
                    self.check_collision_or_terminal_glide(self.game_env, next_row, next_col, row_move_dir=0,
                                                           col_move_dir=move_dir)

                # check if a gem is collected or goal is reached
                next_gem_status, _ = self.check_gem_collected_or_goal_reached(self.game_env, next_row, next_col,
                                                                              state.gem_status)

                if collision:
                    # add any remaining probability to current state
                    res = False, GameState(next_row, next_col, next_gem_status), \
                          reward - self.game_env.collision_penalty, remaining_prob
                    break
                if is_terminal:
                    # add any remaining probability to current state
                    res = False, GameState(next_row, next_col, next_gem_status), \
                          reward - self.game_env.game_over_penalty, remaining_prob
                    break

                # if this state is a possible outcome, add to list
                if d in probs.keys():
                    res = True, GameState(next_row, next_col, next_gem_status), reward, probs[d]
                    remaining_prob -= probs[d]

        elif action in GameEnv.DROP_ACTIONS:
            # drop on any valid tile
            next_row, next_col = state.row, state.col

            drop_amount = {GameEnv.DROP_1: 1, GameEnv.DROP_2: 2, GameEnv.DROP_3: 3}[action]

            # drop until drop amount is reached
            for d in range(1, drop_amount + 1):
                next_row = state.row + d

                # check for collision or game over
                next_row, next_col, collision, is_terminal = \
                    self.check_collision_or_terminal_glide(self.game_env, next_row, next_col, row_move_dir=1,
                                                           col_move_dir=0)

                # check if a gem is collected or goal is reached
                next_gem_status, _ = self.check_gem_collected_or_goal_reached(self.game_env, next_row, next_col,
                                                                              state.gem_status)

                if collision:
                    # add any remaining probability to current state
                    res = False, GameState(next_row, next_col, next_gem_status), \
                          reward - self.game_env.collision_penalty, 1.0
                    break
                if is_terminal:
                    # add any remaining probability to current state
                    res = False, GameState(next_row, next_col, next_gem_status), \
                          reward - self.game_env.game_over_penalty, 1.0
                    break

                if d == drop_amount:
                    res = True, GameState(next_row, next_col, next_gem_status), reward, 1.0
        else:
            assert False, '!!! Invalid action given to perform_action() !!!'

        return res

    def get_transition_outcomes_restricted(self, game_env, state, action):
        """
        This method assumes (state, action) is a valid combination.

        :param game_env: GameEnv instance
        :param state: current state
        :param action: selected action
        :return: list of (next_state, immediate_reward, probability) tuples
        """
        reward = -1 * GameEnv.ACTION_COST[action]
        remaining_prob = 1.0
        outcomes = []

        max_glide1_outcome = max(game_env.glide1_probs.keys())
        max_glide2_outcome = max(game_env.glide2_probs.keys())
        max_glide3_outcome = max(game_env.glide3_probs.keys())

        # handle each action type separately
        if action in GameEnv.WALK_ACTIONS:
            # set movement direction
            if action == GameEnv.WALK_LEFT:
                move_dir = -1
            else:
                move_dir = 1

            # walk on normal walkable tile (super charge case not handled)
            if game_env.grid_data[state.row + 1][state.col] == GameEnv.SUPER_CHARGE_TILE:
                move_dist = game_env.__sample_move_dist(game_env.super_charge_probs)
                next_row, next_col = state.row, state.col
                next_gem_status = state.gem_status

                # move up to the last adjoining supercharge tile
                while game_env.grid_data[next_row + 1][next_col + move_dir] == GameEnv.SUPER_CHARGE_TILE:
                    next_col += move_dir
                    # check for collision or game over
                    next_row, next_col, collision, is_terminal = \
                        self.check_collision_or_terminal(game_env, next_row, next_col,
                                                           row_move_dir=0, col_move_dir=move_dir)
                    if collision or is_terminal:
                        break

                # move sampled move distance beyond the last adjoining supercharge tile
                for d in range(move_dist):
                    next_col += move_dir
                    # check for collision or game over
                    next_row, next_col, collision, is_terminal = \
                        self.check_collision_or_terminal(game_env, next_row, next_col,
                                                           row_move_dir=0, col_move_dir=move_dir)
                    if collision or is_terminal:
                        break

            # if on ladder, handle fall case
            if game_env.grid_data[state.row + 1][state.col] == GameEnv.LADDER_TILE and \
                    game_env.grid_data[state.row + 2][state.col] not in GameEnv.COLLISION_TILES:
                next_row, next_col = state.row + 2, state.col
                # check if a gem is collected or goal is reached
                next_gem_status, _ = self.check_gem_collected_or_goal_reached(game_env, next_row, next_col,
                                                                              state.gem_status)
                outcomes.append((GameState(next_row, next_col, next_gem_status), reward, game_env.ladder_fall_prob,
                                 True))
                remaining_prob -= game_env.ladder_fall_prob

            next_row, next_col = state.row, state.col + move_dir
            # check for collision or game over
            next_row, next_col, collision, is_terminal = \
                self.check_collision_or_terminal(game_env, next_row, next_col, row_move_dir=0, col_move_dir=move_dir)

            # check if a gem is collected or goal is reached
            next_gem_status, is_solved = self.check_gem_collected_or_goal_reached(game_env, next_row, next_col,
                                                                                  state.gem_status)

            if collision:
                # add any remaining probability to current state
                outcomes.append((GameState(next_row, next_col, next_gem_status),
                                 reward - game_env.collision_penalty, remaining_prob, is_solved or is_terminal))
            elif is_terminal:
                # add any remaining probability to current state
                outcomes.append((GameState(next_row, next_col, next_gem_status),
                                 reward - game_env.game_over_penalty, remaining_prob, is_solved or is_terminal))
            else:
                outcomes.append((GameState(next_row, next_col, next_gem_status), reward, remaining_prob,
                                 is_solved or is_terminal))

        elif action == GameEnv.JUMP:
            # jump on normal walkable tile (super jump case not handled)
            if game_env.grid_data[state.row + 1][state.col] == GameEnv.SUPER_JUMP_TILE:
                # sample a random move distance
                move_dist = game_env.__sample_move_dist(game_env.super_jump_probs)

                next_row, next_col = state.row, state.col
                next_gem_status = state.gem_status

                # move sampled distance upwards
                for d in range(move_dist):
                    next_row -= 1
                    # check for collision or game over
                    next_row, next_col, collision, is_terminal = \
                        self.check_collision_or_terminal(game_env, next_row, next_col, reward, row_move_dir=-1, col_move_dir=0)
                    if collision or is_terminal:
                        break
            else:
                next_row, next_col = state.row - 1, state.col
                # check for collision or game over
                next_row, next_col, collision, is_terminal = \
                    self.check_collision_or_terminal(game_env, next_row, next_col, row_move_dir=-1, col_move_dir=0)
            # check if a gem is collected or goal is reached
            next_gem_status, is_solved = self.check_gem_collected_or_goal_reached(game_env, next_row, next_col,
                                                                          state.gem_status)

            if collision:
                # add any remaining probability to current state
                outcomes.append(
                    (GameState(next_row, next_col, next_gem_status), reward - game_env.collision_penalty, 1.0,
                     is_solved or is_terminal))
            elif is_terminal:
                # add any remaining probability to current state
                outcomes.append(
                    (GameState(next_row, next_col, next_gem_status), reward - game_env.game_over_penalty, 1.0,
                     is_solved or is_terminal))
            else:
                outcomes.append((GameState(next_row, next_col, next_gem_status), reward, 1.0, is_solved or is_terminal))

        elif action in GameEnv.GLIDE_ACTIONS:
            # glide on any valid tile
            # select probabilities to sample move distance
            if action in {GameEnv.GLIDE_LEFT_1, GameEnv.GLIDE_RIGHT_1}:
                probs = game_env.glide1_probs
                max_outcome = max_glide1_outcome
            elif action in {GameEnv.GLIDE_LEFT_2, GameEnv.GLIDE_RIGHT_2}:
                probs = game_env.glide2_probs
                max_outcome = max_glide2_outcome
            else:
                probs = game_env.glide3_probs
                max_outcome = max_glide3_outcome

            # set movement direction
            if action in {GameEnv.GLIDE_LEFT_1, GameEnv.GLIDE_LEFT_2, GameEnv.GLIDE_LEFT_3}:
                move_dir = -1
            else:
                move_dir = 1

            # add each possible movement distance to set of outcomes
            next_row, next_col = state.row + 1, state.col
            for d in range(0, max_outcome + 1):
                next_col = state.col + (move_dir * d)
                # check for collision or game over
                next_row, next_col, collision, is_terminal = \
                    self.check_collision_or_terminal_glide(game_env, next_row, next_col, row_move_dir=0,
                                                           col_move_dir=move_dir)

                # check if a gem is collected or goal is reached
                next_gem_status, is_solved = self.check_gem_collected_or_goal_reached(game_env, next_row, next_col,
                                                                              state.gem_status)

                if collision:
                    # add any remaining probability to current state
                    outcomes.append((GameState(next_row, next_col, next_gem_status),
                                     reward - game_env.collision_penalty, remaining_prob, is_solved or is_terminal))
                    break
                if is_terminal:
                    # add any remaining probability to current state
                    outcomes.append((GameState(next_row, next_col, next_gem_status),
                                     reward - game_env.game_over_penalty, remaining_prob, is_solved or is_terminal))
                    break

                # if this state is a possible outcome, add to list
                if d in probs.keys():
                    outcomes.append((GameState(next_row, next_col, next_gem_status), reward, probs[d],
                                     is_solved or is_terminal))
                    remaining_prob -= probs[d]

        elif action in GameEnv.DROP_ACTIONS:
            # drop on any valid tile
            next_row, next_col = state.row, state.col

            drop_amount = {GameEnv.DROP_1: 1, GameEnv.DROP_2: 2, GameEnv.DROP_3: 3}[action]

            # drop until drop amount is reached
            for d in range(1, drop_amount + 1):
                next_row = state.row + d

                # check for collision or game over
                next_row, next_col, collision, is_terminal = \
                    self.check_collision_or_terminal_glide(game_env, next_row, next_col, row_move_dir=1, col_move_dir=0)

                # check if a gem is collected or goal is reached
                next_gem_status, is_solved = self.check_gem_collected_or_goal_reached(game_env, next_row, next_col,
                                                                              state.gem_status)

                if collision:
                    # add any remaining probability to current state
                    outcomes.append((GameState(next_row, next_col, next_gem_status),
                                     reward - game_env.collision_penalty, 1.0, is_solved or is_terminal))
                    break
                if is_terminal:
                    # add any remaining probability to current state
                    outcomes.append((GameState(next_row, next_col, next_gem_status),
                                     reward - game_env.game_over_penalty, 1.0, is_solved or is_terminal))
                    break

                if d == drop_amount:
                    outcomes.append((GameState(next_row, next_col, next_gem_status), reward, 1.0,
                                     is_solved or is_terminal))
                    outcomes.append((GameState(next_row, next_col, next_gem_status), reward, 1.0,
                                     is_solved or is_terminal))

        else:
            assert False, '!!! Invalid action given to perform_action() !!!'

        return outcomes

    def check_collision_or_terminal(self, game_env, row, col, row_move_dir, col_move_dir):
        """
        Checks for collision with solid tile, or entering lava tile. Returns resulting next state (after bounce back if
        colliding), and booleans indicating if collision or game over has occurred.
        :return: (next_row,  next_col, collision (True/False), terminal (True/False))
        """
        terminal = False
        collision = False
        # check for collision condition
        if (not 0 <= row < game_env.n_rows) or (not 0 <= col < game_env.n_cols) or \
                game_env.grid_data[row][col] in GameEnv.COLLISION_TILES:
            row -= row_move_dir  # bounce back to previous position
            col -= col_move_dir  # bounce back to previous position
            collision = True
        # check for game over condition
        elif game_env.grid_data[row][col] == GameEnv.LAVA_TILE:
            terminal = True

        return row, col, collision, terminal

    def check_collision_or_terminal_glide(self, game_env, row, col, row_move_dir, col_move_dir):
        """
        Checks for collision with solid tile, or entering lava tile for the special glide case (player always moves down by
        1, even if collision occurs). Returns resulting next state (after bounce back if colliding), and booleans indicating
        if collision or game over has occurred.
        :return: (next_row,  next_col, collision (True/False), terminal (True/False))
        """
        # variant for checking glide actions - checks row above as well as current row
        terminal = False
        collision = False
        # check for collision condition
        if (not 0 <= row < game_env.n_rows) or (not 0 <= col < game_env.n_cols) or \
                game_env.grid_data[row][col] in GameEnv.COLLISION_TILES or \
                game_env.grid_data[row - 1][col] in GameEnv.COLLISION_TILES:
            row -= row_move_dir  # bounce back to previous position
            col -= col_move_dir  # bounce back to previous position
            collision = True
        # check for game over condition
        elif game_env.grid_data[row][col] == GameEnv.LAVA_TILE or \
                game_env.grid_data[row - 1][col] == GameEnv.LAVA_TILE:
            terminal = True

        return row, col, collision, terminal

    def check_gem_collected_or_goal_reached(self, game_env, row, col, gem_status):
        """
        Checks if the new row and column contains a gem, and returns an updated gem status. Additionally returns a flag
        indicating whether the goal state has been reached (all gems collected and player at exit).
        :return: new gem_status, solved (True/False)
        """
        is_terminal = False
        # check if a gem is collected (only do this for final position of charge)
        if (row, col) in game_env.gem_positions and \
                gem_status[game_env.gem_positions.index((row, col))] == 0:
            gem_status = list(gem_status)
            gem_status[game_env.gem_positions.index((row, col))] = 1
            gem_status = tuple(gem_status)
        # check for goal reached condition (only do this for final position of charge)
        elif row == game_env.exit_row and col == game_env.exit_col and \
                all(gs == 1 for gs in gem_status):
            is_terminal = True
        return gem_status, is_terminal

    def dict_argmax(self, d):
        max_value = max(d.values())
        for k, v in d.items():
            if v == max_value:
                return k

    def run_value_iteration(self, max_time):
        values = {state: 0 for state in self.reachable_states}
        policy = {state: actions[0] for state, actions in self.reachable_actions.items()}
        # policy = {pos: actions[0] for pos, actions in self.reachable_actions.items()}
        for state in self.reachable_states:
            if self.game_env.grid_data[state.row][state.col] == self.game_env.LAVA_TILE:
                values[state] = -self.game_env.game_over_penalty
            if state == self.EXIT_STATE:
                values[state] = 1
        delta = 0
        for s in self.reachable_states:
            action_values = dict()
            print(s)
            if s in policy:
                old_v = values[s]
                r_actions = self.reachable_actions[s]
                print(r_actions)
                for action in r_actions:
                    print(action)
                    outcomes = self.get_transition_outcomes_restricted(self.game_env, s, action)
                    print(outcomes)
                    Q = []
                    for s_next, reward, prob, is_terminal in outcomes:
                        if s_next == self.EXIT_STATE:
                            Q.append(prob * (reward + self.gamma * 0))
                        else:
                            Q.append(prob * (reward + self.gamma * values[s_next]))
                    v = sum(Q)
                    action_values[action] = v
                print(max(action_values.values()))
                print(self.dict_argmax(action_values))
                values[s] = max(action_values.values())
                policy[s] = self.dict_argmax(action_values)
                delta = max(delta, np.abs(old_v - values[s]))
        if delta < self.epsilon:
            self.converged = True
        self.values = values
        self.policy = policy
        # for state, value in values.items():
        #     print("state {}, {} = value of {}".format(state.row, state.col, value))
        #     if state == self.EXIT_STATE:
        #         print("exit state {} {} = value of {}".format(state.row, state.col, value))

    def policy_initialization(self, max_time):
        # t model (lin alg)
        t_model = np.zeros([len(self.reachable_states), len(self.reachable_actions), len(self.reachable_states)])
        for i, s in enumerate(self.reachable_states):
            for j, a in enumerate(self.reachable_actions):
                if s in self.rewards.keys():
                    for k in range(len(self.reachable_states)):
                        if self.reachable_states[k] == (self.game_env.exit_row, self.game_env.exit_col):
                            t_model[i][j][k] = 1.0
                        elif self.game_env.grid_data[self.reachable_states[k].row][self.reachable_states[k].col] == self.game_env.LAVA_TILE:
                            t_model[i][j][k] = -self.game_env.game_over_penalty
                        else:
                            t_model[i][j][k] = 0.0
                elif s == self.EXIT_STATE:
                    t_model[i][j][self.reachable_states.index(self.EXIT_STATE)] = 1.0
                else:
                    reachable_actions = self.reachable_actions[(s.row, s.col)]
                    for action in reachable_actions:
                        # Apply action
                        is_valid, next_state, reward, prob = self.get_transition_probability(s, action)
                        k = self.reachable_states.index(next_state)
                        t_model[i][j][k] += prob
        # r model (lin alg)
        r_model = np.zeros([len(self.reachable_states)])
        for i, s in enumerate(self.reachable_states):
            r_model[i] = self.get_reward(s)
        # lin alg policy
        la_policy = np.zeros([len(self.reachable_states)], dtype=np.int64)
        for i, s in enumerate(self.reachable_states):
            la_policy[i] = 1
            # la_policy[i] = random.randint(0, len(ACTIONS) - 1)
        return t_model, r_model, la_policy

    def run_policy_iteration(self, max_time):
        values = {state: 0 for state in self.reachable_states}
        policy = {pos: np.random.choice(actions) for pos, actions in self.reachable_actions.items()}
        for state in self.reachable_states:
            if self.game_env.grid_data[state.row][state.col] == self.game_env.LAVA_TILE:
                values[state] = -self.game_env.game_over_penalty
            if state == self.EXIT_STATE:
                values[state] = 1
        # policy initialization linear algebra pre-equisites
        t_model, r_model, la_policy = self.policy_initialization(max_time)
        # use linear algebra for policy evaluation
        # V^pi = R + gamma T^pi V^pi
        # (I - gamma * T^pi) V^pi = R
        # Ax = b; A = (I - gamma * T^pi),  b = R
        state_numbers = np.array(range(len(values)))  # indices of every state
        t_pi = t_model[state_numbers, la_policy]
        val_s = np.linalg.solve(np.identity(len(values)) - (self.gamma * t_pi), r_model)
        values = {s: val_s[i] for i, s in enumerate(self.reachable_states)}
        new_policy = {s: self.reachable_actions[(s.row, s.col)][la_policy[i]] for i, s in \
                      enumerate(self.reachable_states)}
        # policy improvement
        for s in self.reachable_states:
            # Keep track of maximum value
            action_values = dict()
            r_actions = self.reachable_actions[(s.row, s.col)]
            for a in r_actions:
                total = 0
                outcomes = self.get_transition_outcomes_restricted(self.game_env, s, a)
                # Apply action
                for next_state, reward, prob, is_terminal in outcomes:
                    if not is_terminal:
                        total += prob * (self.get_reward(s) + (self.gamma * values[next_state]))
                    else:
                        total += 0
                action_values[a] = total
            # Update policy
            new_policy[s] = self.dict_argmax(action_values)

        # Check convergence
        if new_policy == policy:
            self.converged = True
        self.policy = new_policy
        for i, s in enumerate(values):
            la_policy[i] = self._actions.index(policy[(s.row, s.col)])

    # def run_mcts_iteration(self, max_time):

    def plan_offline(self):
        """
        This method will be called once at the beginning of each episode.

        You can use this method to perform value iteration and/or policy iteration and store the computed policy, or
        (optionally) to perform pre-processing for MCTS.

        This planning should not depend on the initial state, as during simulation this is not guaranteed to match the
        initial position listed in the input file (i.e. you may be given a different position to the initial position
        when select_action is called).

        The allowed run time for this method is given by 'game_env.offline_time'. The method will be terminated by the
        simulator if it does not complete within this limit - you should design your algorithm to ensure this method
        exits before the time limit is exceeded.
        """
        t0 = time.time()
        i = 0
        if self.solver_type in {'vi', 'pi', 'mcts'}:
            reach_pos, reach_actions = self.get_reachable_position_and_actions()
            actions = {} #key: state value: actions
            reach_statuses = self.get_reachable_status()
            reachable_state = []
            for i, pos in enumerate(reach_pos):
                for status in reach_statuses:
                    reachable_state.append(GameState(*pos, status))
            for state in reachable_state:
                self.reachable_actions[state] = reach_actions[(state.row, state.col)]
            # self.reachable_actions = reach_actions
            self.reachable_pos = reach_pos
            self.reachable_states = reachable_state
        if self.solver_type == 'vi':
            self.run_value_iteration(self.game_env.offline_time)
        elif self.solver_type == 'pi':
            self.run_policy_iteration(self.game_env.offline_time)
        # optional: loop for ensuring your code exits before the time limit
        # while time.time() - t0 < self.game_env.offline_time or self.converged:
        #     if self.solver_type == 'vi':
        #         self.run_value_iteration(self.game_env.offline_time)
        #     elif self.solver_type == 'pi':
        #         self.run_policy_iteration(self.game_env.offline_time)
        #     i+=1
            # elif self.solver_type =='mcts':
            #     self.res_policy = self.run_mcts_iteration(self.game_env.offline_time)
        t1 = time.time()
        print("Time to complete", i, self.solver_type + " iterations")
        print(t1 - t0)

    def select_action(self, state):
        """
        This method will be called each time the agent is called upon to decide which action to perform (once for each
        step of the episode).

        You can use this to retrieve the optimal action for the current state from a stored offline policy (e.g. from
        value iteration or policy iteration), or to perform MCTS simulations from the current state.

        The allowed run time for this method is given by 'game_env.online_time'. The method will be terminated by the
        simulator if it does not complete within this limit - you should design your algorithm to ensure this method
        exits before the time limit is exceeded.

        :param state: the current state, a GameState instance
        :return: action, the selected action to be performed for the current state
        """
        t0 = time.time()
        i = 0
        # # optional: loop for ensuring your code exits before the time limit
        # while time.time() - t0 < self.game_env.offline_time:
        print("self.policy")
        print(self.policy)
        print("values")
        print(self.values)
        i += 1
        if self.solver_type == 'mcts':
            return self.mcts_search(state)
        if self.solver_type == 'pi':
            return self.policy[state]
        else:
            return self.policy[state]

        t1 = time.time()
        print("Time to complete", i + 1, "PI iterations")
        print(t1 - t0)
