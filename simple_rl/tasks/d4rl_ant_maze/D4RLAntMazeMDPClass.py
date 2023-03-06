import gym
import d4rl
import numpy as np
import random
from copy import deepcopy
import wandb
from pathlib import Path

from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeStateClass import D4RLAntMazeState


class D4RLAntMazeMDP(GoalDirectedMDP):
    def __init__(self, maze_env, goal_state=None, use_hard_coded_events=False, seed=0, render=False):
        assert maze_env in ("antmaze-dynamic-leftmiddle-walls", "antmaze-dynamic-middle-wall", "antmaze-dynamic-rightmiddle-walls"), maze_env

        self.env_name = maze_env

        self.use_hard_coded_events = use_hard_coded_events
        self.record_next = True
        self.total_eps = 0

        self.env = gym.make(self.env_name)
        self.env = gym.wrappers.RecordVideo(self.env, 'video', episode_trigger = lambda x: self._consume_record_next())
        self.reset()

        self.render = render
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        salient_positions = []

        if use_hard_coded_events:
            salient_positions = [np.array((4, 0)),
                                 np.array((8, 0)),
                                 np.array((8, 4)),
                                 np.array((8, 8)),
                                 np.array((4, 8)),
                                 np.array((0, 8))]

        self.dataset = self.env.get_dataset()["observations"]
        self._determine_x_y_lims()

        GoalDirectedMDP.__init__(self, range(self.env.action_space.shape[0]),
                                 self._transition_func,
                                 self._reward_func, self.init_state,
                                 salient_positions, task_agnostic=goal_state is None,
                                 goal_state=goal_state, goal_tolerance=0.6)

    def _consume_record_next(self):
        if self.record_next:
            self.record_next = False
            return True
        return False

    def record_next_ep(self):
        self.record_next = True

    def _reward_func(self, state, action):
        next_state, _, done, info = self.env.step(action)

        time_limit_truncated = info.get('TimeLimit.truncated', False)
        is_terminal = done and not time_limit_truncated

        if self.task_agnostic:  # No reward function => no rewards and no terminations
            reward = 0.
            is_terminal = False
        else:
            reward, is_terminal = self.sparse_gc_reward_function(next_state, self.get_current_goal(), {}) # TODO info

        if self.render:
            self.env.render()

        self.next_state = self._get_state(next_state, is_terminal)

        return reward

    def _transition_func(self, state, action):
        return self.next_state

    def _get_state(self, observation, done):
        """ Convert np obs array from gym into a State object. """
        obs = np.copy(observation)
        position = obs[:2]
        others = obs[2:]
        state = D4RLAntMazeState(position, others, done)
        return state

    def state_space_size(self):
        return self.env.observation_space.shape[0]

    def action_space_size(self):
        return self.env.action_space.shape[0]

    def get_init_positions(self):
        return [self.init_state.position]

    def switch_environment(self, new_env):
        assert new_env in ("antmaze-dynamic-leftmiddle-walls", "antmaze-dynamic-middle-wall", "antmaze-dynamic-rightmiddle-walls"), new_env
        self.env = gym.make(new_env)
        self.env = gym.wrappers.RecordVideo(self.env, 'video', episode_trigger = lambda x: self._consume_record_next())
        self.env_name = new_env
        self.env.seed(self.seed)
        self.env.reset()

    def reset(self, episode=None):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)

        p = f"video/rl-video-episode-{self.total_eps-10}.mp4"
        if Path(p).exists():
            wandb.log({"test_videos": wandb.Video(p)})
        self.total_eps += 1

        super(D4RLAntMazeMDP, self).reset()


    def set_xy(self, position):
        """ Used at test-time only. """
        position = tuple(position)  # `maze_model.py` expects a tuple
        self.env.env.set_xy(position)
        obs = np.concatenate((np.array(position), self.init_state.features()[2:]), axis=0)
        self.cur_state = self._get_state(obs, done=False)
        self.init_state = deepcopy(self.cur_state)

    # --------------------------------
    # Used for visualizations only
    # --------------------------------

    def _determine_x_y_lims(self):
        observations = self.dataset
        x = [obs[0] for obs in observations]
        y = [obs[1] for obs in observations]
        xlow, xhigh = min(x), max(x)
        ylow, yhigh = min(y), max(y)
        self.xlims = (xlow, xhigh)
        self.ylims = (ylow, yhigh)

    def get_x_y_low_lims(self):
        return self.xlims[0], self.ylims[0]

    def get_x_y_high_lims(self):
        return self.xlims[1], self.ylims[1]

    # ---------------------------------
    # Used during testing only
    # ---------------------------------

    def sample_random_state(self, cond=lambda x: True):
        num_tries = 0
        rejected = True
        while rejected and num_tries < 200:
            low = np.array((self.xlims[0], self.ylims[0]))
            high = np.array((self.xlims[1], self.ylims[1]))
            sampled_point = np.random.uniform(low=low, high=high)
            rejected = self.env.env.wrapped_env._is_in_collision(sampled_point) or not cond(sampled_point)
            num_tries += 1

            if not rejected:
                return sampled_point

    def sample_random_action(self):
        size = (self.action_space_size(),)
        return np.random.uniform(-1., 1., size=size)
    
    def sample_random_state_curriculum(self, x, y):
        """
        Samples points in the rectangle whose (x,y) is
        the lower left corner (Jason)
        """
        data = self.dataset
        mask = (data[:,0] >= x) & (data[:,1] >= y)
        filtered_data = data[mask]
        idx = np.random.choice(filtered_data.shape[0])
        return filtered_data[idx, :]
