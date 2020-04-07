# Python imports.
import numpy as np
import random
import ipdb
from scipy.spatial import distance

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.point_reacher.PointReacherStateClass import PointReacherState
from simple_rl.tasks.point_maze.environments.point_maze_env import PointMazeEnv
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent


class PointReacherMDP(MDP):
    def __init__(self, seed, color_str="", dense_reward=False, render=False):
        self.env_name = "point_reacher"
        self.seed = seed
        self.dense_reward = dense_reward
        self.render = render

        random.seed(seed)
        np.random.seed(seed)

        # Configure env
        gym_mujoco_kwargs = {
            'maze_id': 'Reacher',
            'n_bins': 0,
            'observe_blocks': False,
            'put_spin_near_agent': False,
            'top_down_view': False,
            'manual_collision': True,
            'maze_size_scaling': 3,
            'color_str': color_str
        }
        self.env = PointMazeEnv(**gym_mujoco_kwargs)
        self.reset()

        self.salient_positions = [np.array((-9, 9)), np.array((9, 9))]
                                  # np.array((9, -9)), np.array((-9, -9))]

        # Set the current target events in the MDP
        self.current_salient_events = [SalientEvent(pos) for pos in self.salient_positions]

        # Set an ever expanding list of salient events - we need to keep this around to call is_term_true on trained options
        self.original_salient_events = [SalientEvent(pos) for pos in self.salient_positions]

        # In some MDPs, we use a predicate to determine if we are at the start state of the MDP
        self.start_state_salient_event = SalientEvent(target_state=self.init_state.position)

        MDP.__init__(self, [1, 2], self._transition_func, self._reward_func, self.init_state)

    def _reward_func(self, state, action):
        next_state, reward, done, _ = self.env.step(action)
        if self.render:
            self.env.render()
        self.next_state = self._get_state(next_state, done)
        if self.dense_reward:
            raise NotImplementedError("Dense rewards for point-reacher")
        return reward + 1  # TODO: Changing the reward function to return 0 step penalty and 1 reward

    def _transition_func(self, state, action):
        return self.next_state

    def _get_state(self, observation, done):
        """ Convert np obs array from gym into a State object. """  # TODO: Adapt has_key
        obs = np.copy(observation)
        position = obs[:2]
        has_key = obs[2]
        theta = obs[3]
        velocity = obs[4:6]
        theta_dot = obs[6]
        # Ignoring obs[7] which corresponds to time elapsed in seconds
        state = PointReacherState(position, theta, velocity, theta_dot, done)
        return state

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(PointReacherMDP, self).execute_agent_action(action)
        return reward, next_state

    def get_current_target_events(self):
        """ Return list of predicate functions that indicate salience in this MDP. """
        return self.current_salient_events

    def get_original_target_events(self):
        return self.original_salient_events

    def is_goal_state(self, state):
        # return any([predicate(state) for predicate in self.get_current_target_events()])
        return False

    def is_start_state(self, state):
        pos = self._get_position(state)
        s0 = self.init_state.position
        return np.linalg.norm(pos - s0) <= 0.6

    def batched_is_start_state(self, position_matrix):
        s0 = self.init_state.position
        in_start_pos = distance.cdist(position_matrix, s0[None, :]) <= 0.6
        return in_start_pos.squeeze(1)

    def get_start_state_salient_event(self):
        return self.start_state_salient_event

    def satisfy_target_event(self, option):
        """
        Once a salient event has both forward and backward options related to it,
        we no longer need to maintain it as a target_event. This function will find
        the salient event that corresponds to the input state and will remove that
        event from the list of target_events.

        Args:
            option (Option)

        Returns:

        """
        if option.backward_option:
            for salient_event in self.current_salient_events:
                satisfied_salience = option.is_init_true(salient_event.target_state)

                if satisfied_salience and (salient_event in self.current_salient_events):
                    self.current_salient_events.remove(salient_event)

    @staticmethod
    def _get_position(state):
        position = state.position if isinstance(state, PointReacherState) else state[:2]
        return position

    @staticmethod
    def state_space_size():
        return 6

    @staticmethod
    def action_space_size():
        return 2

    @staticmethod
    def is_primitive_action(action):
        return -1. <= action.all() <= 1.

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(PointReacherMDP, self).reset()

    def set_xy(self, position):
        self.env.wrapped_env.set_xy(position)
        self.cur_state = self._get_state(np.array((position[0], position[1], 0, 0, 0, 0, 0)), done=False)

    def __str__(self):
        return self.env_name
