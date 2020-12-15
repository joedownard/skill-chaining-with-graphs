import pickle

import torch
import numpy as np
import torch.nn as nn

from copy import deepcopy
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from simple_rl.agents.func_approx.dsc.dynamics.dynamics_model import DynamicsModel
from simple_rl.agents.func_approx.dsc.dynamics.replay_buffer import ReplayBuffer

from tqdm import tqdm
import ipdb

class MPC:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.is_trained = False
        self.trained_options = []
        self.gamma = 0.99

        self.model = DynamicsModel(self.state_size, self.action_size, self.device)
        self.model.to(self.device)
        
        self.replay_buffer = ReplayBuffer(obs_dim=state_size, act_dim=action_size, size=int(3e5))

    def load_data(self):
        self.dataset = self._preprocess_data()
        self.model.set_standardization_vars(*self._get_standardization_vars())

    def train(self, epochs=100, batch_size=512):
        self.is_trained = True

        training_gen = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        loss_function = nn.MSELoss().to(self.device)
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        
        for epoch in tqdm(range(epochs), desc=f'Training MPC model on {self.replay_buffer.size} points'):
            for states, actions, states_p in training_gen:
                states = states.to(self.device).float()
                actions = actions.to(self.device).float()
                states_p = states_p.to(self.device).float()
            
                optimizer.zero_grad()
                p = self.model.forward(states, actions)
                loss = loss_function(p, states_p)
                loss.backward()
                optimizer.step()

    def rollout(self, mdp, num_rollouts, num_steps, goal, max_steps):
        steps_taken = 0
        s = deepcopy(mdp.cur_state)

        while not mdp.sparse_gc_reward_function(s, goal, {})[1]:
            action = self.act(s, goal, num_rollouts, num_steps)
        
            # execute action in mdp
            mdp.execute_agent_action(action)

            steps_taken += 1
            if steps_taken == max_steps:
                break

            # retrieve current state
            s = deepcopy(mdp.cur_state)

        return deepcopy(mdp.cur_state), steps_taken

    def _rollout_debug(self, mdp, num_rollouts, num_steps, goal, max_steps, gamma=0.95):
        steps_taken = 0
        s = deepcopy(mdp.cur_state)

        trajectory = [s]

        while not mdp.sparse_gc_reward_function(s, goal, {})[1]:
            action = self.act(s, goal, num_rollouts, num_steps, gamma=gamma)
        
            # execute action in mdp
            mdp.execute_agent_action(action)

            steps_taken += 1
            if steps_taken == max_steps:
                break

            # retrieve current state
            s = deepcopy(mdp.cur_state)

            trajectory.append(s)

        return deepcopy(mdp.cur_state), steps_taken, trajectory

    def act(self, s, goal, num_rollouts=14000, num_steps=7, gamma=0.95):
        # sample actions for all steps
        goal_x = goal[0]
        goal_y = goal[1]
        np_actions = np.random.uniform(-1., 1., size=(num_rollouts, num_steps, self.action_size)) # TODO hardcoded
        np_states = np.repeat(np.array([s]), num_rollouts, axis=0)
        results = np.zeros((num_rollouts, num_steps))
        
        with torch.no_grad():
            # compute next states for each step
            for j in range(num_steps):
                actions = np_actions[:,j,:]
                states_t = torch.from_numpy(np_states)
                actions_t = torch.from_numpy(actions)

                # transfer to gpu
                states_t = states_t.to(self.device)
                actions_t = actions_t.to(self.device)

                pred = self.model.predict_next_state(states_t.float(), actions_t.float())
                np_states = pred.cpu().numpy()

                # update results with (any) distance metric
                results[:,j] = (goal_x - np_states[:,0]) ** 2 + (goal_y - np_states[:,1]) ** 2
        
        # choose next action to execute
        gammas = np.power(gamma * np.ones(num_steps), np.arange(0, num_steps))
        summed_results = np.sum(results * gammas, axis=1)
        index = np.argmin(summed_results) # retrieve action with least trajectory distance to goal
        action = np_actions[index,0,:] # grab action corresponding to least distance
        return action

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.store(state, action, reward, next_state, done)

    def _preprocess_data(self):
        states = self.replay_buffer.obs_buf[:self.replay_buffer.size, :]
        actions = self.replay_buffer.act_buf[:self.replay_buffer.size, :]
        states_p = self.replay_buffer.obs2_buf[:self.replay_buffer.size, :]

        assert states.shape[1] == states_p.shape[1] == self.state_size, f"{states.shape, states_p.shape}"
        assert actions.shape[1] == self.action_size, f"{actions.shape}"

        states_delta = np.array(states_p) - np.array(states)
        
        self.mean_x = np.mean(states, axis=0)
        self.mean_y = np.mean(actions, axis=0)
        self.mean_z = np.mean(states_delta, axis=0)
        self.std_x = np.std(states - self.mean_x, axis=0)
        self.std_y = np.std(actions - self.mean_y, axis=0)
        self.std_z = np.std(states_delta - self.mean_z, axis=0)

        self._roundup()

        norm_states_delta = (states_delta - self.mean_z) / self.std_z

        dataset = RolloutDataset(states, actions, norm_states_delta)
        return dataset

    def _roundup(self, c=1e-5):
        """
        If any standarization variable is 0, add some constant to prevent NaN
        """
        self.std_x[self.std_x == 0] = c
        self.std_y[self.std_y == 0] = c
        self.std_z[self.std_z == 0] = c

    def _get_standardization_vars(self):
        return self.mean_x, self.mean_y, self.mean_z, self.std_x, self.std_y, self.std_z

    def save_model(self, path):
        if self.model is not None:
            state_dictionary = self.model.__getstate__()
            with open(path, 'wb') as f:
                pickle.dump(state_dictionary, f)
        else:
            print("no model has been trained yet!")

    def load_model(self, path):
        with open(path, 'rb') as f:
            state_dictionary = pickle.load(f)
        self.model = DynamicsModel(self.state_size, self.action_size, self.device)
        self.model.__setstate__(state_dictionary)

class RolloutDataset(Dataset):
    def __init__(self, states, actions, states_p):
        self.states = states
        self.actions = actions
        self.states_p = states_p
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.states_p[idx]