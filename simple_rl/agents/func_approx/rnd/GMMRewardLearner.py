import time
import numpy as np
from collections import deque
from sklearn.mixture import GaussianMixture
from simple_rl.agents.func_approx.rnd.utils import RunningMeanStd


class GMM:
    def __init__(self, use_reward_norm, update_interval, use_position_subset, buffer_size):
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.use_reward_norm = use_reward_norm
        self.use_position_subset = use_position_subset

        self.memory = deque([], maxlen=buffer_size)
        self.gmm = GaussianMixture(n_components=8, warm_start=True)

        if use_reward_norm:
            self.reward_rms = RunningMeanStd()

        self.name = "gmm-reward-module"

    def get_reward(self, states):
        assert isinstance(states, np.ndarray), f"{type(states)}"

        if self.use_position_subset:
            states = self.get_state_subset(states)

        log_probabilities = self.gmm.score_samples(states)
        rewards = -1. * log_probabilities

        if self.use_reward_norm:
            rewards /= np.sqrt(self.reward_rms.var)

        return rewards

    def update(self, state):
        assert isinstance(state, np.ndarray), state

        if self.use_position_subset:
            state = self.get_state_subset(state)

        self.memory.append(state)

        if len(self.memory) >= self.update_interval:
            states = np.array(self.memory)
            self.train(states)

    def train(self, states):
        assert isinstance(states, np.ndarray), f"{type(states)}"
        assert len(states.shape) == 2, f"Expected (batch, state_dim), got {states.shape}"

        if self.use_position_subset:
            assert states.shape[1] == 2, states.shape

        t0 = time.time()
        self.gmm.fit(states)
        dt = np.round(time.time()-t0, decimals=2)
        print(f"Trained GMM on data of shape {states.shape} ({dt}s)")

    def update_reward_rms(self, episodic_rewards):
        assert self.use_reward_norm, f"use_reward_norm={self.use_reward_norm}"

        if len(episodic_rewards) > 0:
            mean = np.mean(episodic_rewards)
            std = np.std(episodic_rewards)
            size = len(episodic_rewards)

            self.reward_rms.update_from_moments(mean, std ** 2, size)

    @staticmethod
    def get_state_subset(states):
        assert isinstance(states, np.ndarray), f"{type(states)}"

        if len(states.shape) == 1:
            return states[:2]
        if states.shape[0] == 1:
            return states[0, :2]
        return states[:, :2]
