# Python imports.
from __future__ import print_function

import sys
sys.path = [""] + sys.path

from collections import deque, defaultdict
from copy import deepcopy
import ipdb
import argparse
import os
import random
import numpy as np
from tensorboardX import SummaryWriter
import torch
import itertools
from tqdm import tqdm

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.agents.func_approx.ddpg.utils import *
from simple_rl.agents.func_approx.exploration.utils import *
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain
from simple_rl.agents.func_approx.dsc.GraphSearchClass import GraphSearch
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent


class SkillChaining(object):
	def __init__(self, mdp, max_steps, lr_actor, lr_critic, ddpg_batch_size, device, max_num_options=5,
				 subgoal_reward=0., enable_option_timeout=True, buffer_length=20, num_subgoal_hits_required=3,
				 classifier_type="ocsvm", init_q=None, generate_plots=False, use_full_smdp_update=False,
				 start_state_salience=False, log_dir="", seed=0, tensor_log=False):
		"""
		Args:
			mdp (MDP): Underlying domain we have to solve
			max_steps (int): Number of time steps allowed per episode of the MDP
			lr_actor (float): Learning rate for DDPG Actor
			lr_critic (float): Learning rate for DDPG Critic
			ddpg_batch_size (int): Batch size for DDPG agents
			device (str): torch device {cpu/cuda:0/cuda:1}
			subgoal_reward (float): Hitting a subgoal must yield a supplementary reward to enable local policy
			enable_option_timeout (bool): whether or not the option times out after some number of steps
			buffer_length (int): size of trajectories used to train initiation sets of options
			num_subgoal_hits_required (int): number of times we need to hit an option's termination before learning
			classifier_type (str): Type of classifier we will train for option initiation sets
			init_q (float): If not none, we use this value to initialize the value of a new option
			generate_plots (bool): whether or not to produce plots in this run
			use_full_smdp_update (bool): sparse 0/1 reward or discounted SMDP reward for training policy over options
			start_state_salience (bool): Treat the start state of the MDP as a salient event OR create intersection events
			log_dir (os.path): directory to store all the scores for this run
			seed (int): We are going to use the same random seed for all the DQN solvers
			tensor_log (bool): Tensorboard logging enable
		"""
		self.mdp = mdp
		self.original_actions = deepcopy(mdp.actions)
		self.max_steps = max_steps
		self.subgoal_reward = subgoal_reward
		self.enable_option_timeout = enable_option_timeout
		self.init_q = init_q
		self.use_full_smdp_update = use_full_smdp_update
		self.generate_plots = generate_plots
		self.buffer_length = buffer_length
		self.num_subgoal_hits_required = num_subgoal_hits_required
		self.log_dir = log_dir
		self.seed = seed
		self.device = torch.device(device)
		self.max_num_options = max_num_options
		self.classifier_type = classifier_type
		self.dense_reward = mdp.dense_reward
		self.start_state_salience = start_state_salience

		tensor_name = "runs/{}_{}".format(args.experiment_name, seed)
		self.writer = SummaryWriter(tensor_name) if tensor_log else None

		print("Initializing skill chaining with option_timeout={}, seed={}".format(self.enable_option_timeout, seed))

		random.seed(seed)
		np.random.seed(seed)

		self.validation_scores = []

		# This option has an initiation set that is true everywhere and is allowed to operate on atomic timescale only
		self.global_option = Option(overall_mdp=self.mdp, name="global_option", global_solver=None,
									lr_actor=lr_actor, lr_critic=lr_critic, buffer_length=buffer_length,
									ddpg_batch_size=ddpg_batch_size, num_subgoal_hits_required=num_subgoal_hits_required,
									subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
									enable_timeout=self.enable_option_timeout, classifier_type=classifier_type,
									generate_plots=self.generate_plots, writer=self.writer, device=self.device,
									dense_reward=self.dense_reward, chain_id=None, is_backward_option=False, option_idx=0)

		self.trained_options = [self.global_option]

		# Pointer to the current option:
		# 1. This option has the termination set which defines our current goal trigger
		# 2. This option has an untrained initialization set and policy, which we need to train from experience
		self.untrained_options = []

		# This is our first untrained option - one that gets us to the goal state from nearby the goal
		# We pick this option when self.agent_over_options thinks that we should
		# Once we pick this option, we will use its internal DDPG solver to take primitive actions until termination
		# Once we hit its termination condition N times, we will start learning its initiation set
		# Once we have learned its initiation set, we will create its child option
		for i, salient_event in enumerate(self.mdp.get_original_target_events()):
			goal_option = Option(overall_mdp=self.mdp, name=f'goal_option_{i + 1}', global_solver=self.global_option.solver,
								 lr_actor=lr_actor, lr_critic=lr_critic, buffer_length=buffer_length,
								 ddpg_batch_size=ddpg_batch_size, num_subgoal_hits_required=num_subgoal_hits_required,
								 subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
								 enable_timeout=self.enable_option_timeout, classifier_type=classifier_type,
								 generate_plots=self.generate_plots, writer=self.writer, device=self.device,
								 dense_reward=self.dense_reward, chain_id=i+1, max_num_children=1,
								 target_salient_event=salient_event, option_idx=i+1)
			self.untrained_options.append(goal_option)

		# This is our policy over options
		# We use (double-deep) (intra-option) Q-learning to learn the Q-values of *options* at any queried state Q(s, o)
		# We start with this DQN Agent only predicting Q-values for taking the global_option, but as we learn new
		# options, this agent will predict Q-values for them as well
		self.agent_over_options = DQNAgent(self.mdp.state_space_size(), 1, trained_options=self.trained_options,
										   seed=seed, lr=1e-4, name="GlobalDQN", eps_start=1.0, tensor_log=tensor_log,
										   use_double_dqn=True, writer=self.writer, device=self.device,
										   exploration_strategy="shaping")

		# Keep track of which chain each created option belongs to
		s0 = self.mdp.env._init_positions
		self.s0 = s0
		self.chains = [SkillChain(start_states=s0, options=[], chain_id=(i+1),
		 			   			  intersecting_options=[], mdp_start_states=s0, is_backward_chain=False,
								  target_salient_event=salient_event)
					   for i, salient_event in enumerate(self.mdp.get_original_target_events())]

		# List of init states seen while running this algorithm
		self.init_states = []

		self.current_option_idx = 1
		self.current_intersecting_chains = []

		# This is our temporally extended exploration option
		# We use it to drive the agent towards unseen parts of the state-space
		# The temporally extended nature of the option allows the agent to solve the "option sink" problem
		# Note that we use a smaller timeout for this option so as to permit creating new options after it
		# target_predicate = lambda s: goal_option_1.is_term_true(s) or goal_option_2.is_term_true(s)
		# batched_target_predicate = lambda s: np.logical_or(goal_option_1.batched_is_term_true(s),
		# 												   goal_option_2.batched_is_term_true(s))
		# exploration_option = Option(overall_mdp=self.mdp, name="exploration_option", global_solver=self.global_option.solver,
		# 							lr_actor=self.global_option.solver.actor_learning_rate,
		# 							lr_critic=self.global_option.solver.critic_learning_rate,
		# 							buffer_length=self.global_option.buffer_length,
		# 							ddpg_batch_size=self.global_option.solver.batch_size,
		# 							num_subgoal_hits_required=self.num_subgoal_hits_required,
		# 							subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps,
		# 							enable_timeout=self.enable_option_timeout, classifier_type=self.classifier_type,
		# 							generate_plots=self.generate_plots, writer=self.writer, device=self.device,
		# 							dense_reward=self.dense_reward, chain_id=None, timeout=30,
		# 							init_predicate=target_predicate, batched_init_predicate=batched_target_predicate)

		# # Add the exploration option to the policy over options
		# self._augment_agent_with_new_option(exploration_option, 0.)

		# If initialize_everywhere is True, also add to trained_options
		for option in self.untrained_options:  # type: Option
			if option.initialize_everywhere:
				self._augment_agent_with_new_option(option, 0.)

		# Debug variables
		self.global_execution_states = []
		self.num_option_executions = defaultdict(lambda : [])
		self.option_rewards = defaultdict(lambda : [])
		self.option_qvalues = defaultdict(lambda : [])
		self.num_options_history = []

	def add_negative_examples(self, new_option):
		""" When creating an option in the forward chain, add the positive examples of all the options
			in the forward chain as negative examples for the current option. Do the same when creating
			a new_option in the backward chain, i.e, only add positives in the backward chain as
			negative examples for the new_option.
		"""
		forward_option = not new_option.backward_option
		if forward_option:
			for option in self.trained_options[1:]:  # type: Option
				new_option.negative_examples += option.positive_examples
		else:
			for option in self.trained_options[1:]:  # type: Option
				if option.backward_option:
					new_option.negative_examples += option.positive_examples

	def init_state_in_option(self, option):
		"""
		If the input `option` is part of a forward chain, we want to check if the start
		state of the MDP is in its initiation set. If so, we will not create a child
		option for the input `option`. If the input `option` is part of the backward
		chain, we want to check that ANY of the chain start states (usually existing
		salient events and optionally the start state of the MDP) are if in its
		initiation set classifier.
		Args:
			option (Option): We want to check if we want to create a child for this option

		Returns:
			any_init_in_option (bool)
		"""
		start_states = self.chains[option.chain_id - 1].start_states
		starts_chained = [option.is_init_true(state) for state in start_states]
		return any(starts_chained)

	def state_in_any_option(self, state):
		for option in self.trained_options[1:]:
			if option.is_init_true(state):
				return True
		return False

	def create_children_options(self, option):
		o1 = self.create_child_option(option)
		children = [o1]
		current_parent = option.parent
		while current_parent is not None:
			child_option = self.create_child_option(current_parent)
			children.append(child_option)
			current_parent = current_parent.parent
		return children

	def create_child_option(self, parent_option):

		# Don't create a child option if you already include the init state of the MDP
		# Also enforce the max branching factor of our skill tree

		# Added extra condition for chain_id b/c I got an intersection option which covered the start state,
		# but still want options to build in the other direction
		# if (self.init_state_in_option(parent_option) and parent_option.name != "intersection_option") \
		if self.init_state_in_option(parent_option)	\
				or len(parent_option.children) >= parent_option.max_num_children:
			return None

		# Create new option whose termination is the initiation of the option we just trained
		if "goal_option" in parent_option.name:
			prefix = f"option_{parent_option.name.split('_')[-1]}"
		else:
			prefix = parent_option.name
		name = prefix + "_{}".format(len(parent_option.children))
		print("Creating {} with parent {}".format(name, parent_option.name))

		# Options in the backward chain have a pre-set init_salient_event defined for them.
		# This is generally the target event of the corresponding forward skill chain. Options
		# in the current backward chain have their init_salient_event parameter set, which
		# is only used during the gestation period of options
		is_backward_option = self.chains[parent_option.chain_id - 1].is_backward_chain
		gestation_init_salient_event = None
		if is_backward_option:
			gestation_init_salient_event = parent_option.init_salient_event  # type: SalientEvent

		old_untrained_option_id = id(parent_option)
		new_untrained_option = Option(self.mdp, name=name, global_solver=self.global_option.solver,
									  lr_actor=parent_option.solver.actor_learning_rate,
									  lr_critic=parent_option.solver.critic_learning_rate,
									  ddpg_batch_size=parent_option.solver.batch_size,
									  subgoal_reward=self.subgoal_reward,
									  buffer_length=self.buffer_length,
									  classifier_type=self.classifier_type,
									  num_subgoal_hits_required=self.num_subgoal_hits_required,
									  seed=self.seed, parent=parent_option,  max_steps=self.max_steps,
									  enable_timeout=self.enable_option_timeout, chain_id=parent_option.chain_id,
									  writer=self.writer, device=self.device, dense_reward=self.dense_reward,
									  initialize_everywhere=parent_option.initialize_everywhere,
									  max_num_children=parent_option.max_num_children,
									  is_backward_option=is_backward_option,
									  init_salient_event=gestation_init_salient_event)

		new_untrained_option_id = id(new_untrained_option)
		assert new_untrained_option_id != old_untrained_option_id, "Checking python references"
		assert id(new_untrained_option.parent) == old_untrained_option_id, "Checking python references"

		parent_option.children.append(new_untrained_option)

		return new_untrained_option

	def make_off_policy_updates_for_options(self, state, action, reward, next_state):
		for option in self.trained_options: # type: Option
			option.off_policy_update(state, action, reward, next_state)\


	def make_smdp_update(self, state, action, total_discounted_reward, next_state, option_transitions):
		"""
		Use Intra-Option Learning for sample efficient learning of the option-value function Q(s, o)
		Args:
			state (State): state from which we started option execution
			action (int): option taken by the global solver
			total_discounted_reward (float): cumulative reward from the overall SMDP update
			next_state (State): state we landed in after executing the option
			option_transitions (list): list of (s, a, r, s') tuples representing the trajectory during option execution
		"""

		def get_reward(transitions):
			gamma = self.global_option.solver.gamma
			raw_rewards = [tt[2] for tt in transitions]
			return sum([(gamma ** idx) * rr for idx, rr in enumerate(raw_rewards)])

		selected_option = self.trained_options[action]  # type: Option
		for i, transition in enumerate(option_transitions):
			start_state = transition[0]
			if selected_option.is_init_true(start_state):
				if self.use_full_smdp_update:
					sub_transitions = option_transitions[i:]
					option_reward = get_reward(sub_transitions)
					self.agent_over_options.step(start_state.features(), action, option_reward, next_state.features(),
												 next_state.is_terminal(), num_steps=len(sub_transitions))
				else:
					option_reward = self.subgoal_reward if selected_option.is_term_true(next_state) else -1.
					self.agent_over_options.step(start_state.features(), action, option_reward, next_state.features(),
												 next_state.is_terminal(), num_steps=1)

	def get_init_q_value_for_new_option(self, newly_trained_option):
		global_solver = self.agent_over_options  # type: DQNAgent
		state_option_pairs = newly_trained_option.final_transitions
		q_values = []
		for state, option_idx in state_option_pairs:
			q_value = global_solver.get_qvalue(state.features(), option_idx)
			q_values.append(q_value)
		return np.max(q_values)

	def _augment_agent_with_new_option(self, newly_trained_option, init_q_value):
		"""
		Train the current untrained option and initialize a new one to target.
		Add the newly_trained_option as a new node to the Q-function over options
		Args:
			newly_trained_option (Option)
			init_q_value (float): if given use this, else compute init_q optimistically
		"""
		# Add the trained option to the action set of the global solver
		if newly_trained_option not in self.trained_options:
			self.trained_options.append(newly_trained_option)
			self.global_option.solver.trained_options.append(newly_trained_option)

		# Add the newly trained option to the corresponding skill chain
		if newly_trained_option.chain_id is not None:
			chain_idx = newly_trained_option.chain_id - 1
			self.chains[chain_idx].options.append(newly_trained_option)

		# Set the option_idx for the newly added option
		newly_trained_option.option_idx = self.current_option_idx
		self.current_option_idx += 1

		# Augment the global DQN with the newly trained option
		num_actions = len(self.trained_options)
		new_global_agent = DQNAgent(self.agent_over_options.state_size, num_actions, self.trained_options,
									seed=self.seed, name=self.agent_over_options.name,
									eps_start=self.agent_over_options.epsilon,
									tensor_log=self.agent_over_options.tensor_log,
									use_double_dqn=self.agent_over_options.use_ddqn,
									lr=self.agent_over_options.learning_rate,
									writer=self.writer, device=self.device,
									exploration_strategy="shaping")
		new_global_agent.replay_buffer = self.agent_over_options.replay_buffer

		init_q = self.get_init_q_value_for_new_option(newly_trained_option) if init_q_value is None else init_q_value
		print("Initializing new option node with q value {}".format(init_q))
		new_global_agent.policy_network.initialize_with_smaller_network(self.agent_over_options.policy_network, init_q)
		new_global_agent.target_network.initialize_with_smaller_network(self.agent_over_options.target_network, init_q)

		self.agent_over_options = new_global_agent

	def initialize_policy_over_options(self):
		""" Can be called after call to _augment_(). Useful if we want to reset the policy over options. """
		num_actions = len(self.trained_options)
		new_global_agent = DQNAgent(self.agent_over_options.state_size, num_actions, self.trained_options,
									seed=self.seed, name=self.agent_over_options.name,
									eps_start=self.agent_over_options.epsilon,
									tensor_log=self.agent_over_options.tensor_log,
									use_double_dqn=self.agent_over_options.use_ddqn,
									lr=self.agent_over_options.learning_rate,
									writer=self.writer, device=self.device,
									exploration_strategy="shaping")
		self.agent_over_options = new_global_agent

	def _get_backward_options_in_gestation(self, state):
		for option in self.trained_options:
			if option.backward_option and option.get_training_phase() == "gestation" and \
					option.is_init_true(state) and not option.is_term_true(state):
				return option
		return None

	def act(self, state):
		# Query the global Q-function to determine which option to take in the current state
		option_idx = self.agent_over_options.act(state.features(), train_mode=True)
		self.agent_over_options.update_epsilon()

		# Selected option
		selected_option = self.trained_options[option_idx]  # type: Option

		return selected_option

	def take_action(self, state, episode_number, step_number):
		"""
		Either take a primitive action from `state` or execute a closed-loop option policy.
		Args:
			state (State)
			episode_number (int)
			step_number (int): which iteration of the control loop we are on

		Returns:
			experiences (list): list of (s, a, r, s') tuples
			reward (float): sum of all rewards accumulated while executing chosen action
			next_state (State): state we landed in after executing chosen action
		"""
		selected_option = self.act(state)

		# TODO: Hack - If you satisfy the initiation condition for a backward option in gestation, take it
		backward_option = self._get_backward_options_in_gestation(state)
		if backward_option is not None:
			selected_option = backward_option

		# TODO: Hack - do not take option from a salient event targeting it
		if selected_option.is_term_true(state):
			selected_option = self.global_option

		option_transitions, discounted_reward = selected_option.execute_option_in_mdp(self.mdp, episode_number, step_number)

		option_reward = self.get_reward_from_experiences(option_transitions)
		next_state = self.get_next_state_from_experiences(option_transitions)

		# If we triggered the untrained option's termination condition, add to its buffer of terminal transitions
		for untrained_option in self.untrained_options:
			if untrained_option.is_term_true(next_state) and not untrained_option.is_term_true(state):
				untrained_option.final_transitions.append((state, selected_option.option_idx))

		# Add data to train Q(s, o)
		self.make_smdp_update(state, selected_option.option_idx, discounted_reward, next_state, option_transitions)

		return option_transitions, option_reward, next_state, len(option_transitions)

	def sample_qvalue(self, option):
		""" Sample Q-value from the given option's intra-option policy. """
		if len(option.solver.replay_buffer) > 500:
			sample_experiences = option.solver.replay_buffer.sample(batch_size=500)
			sample_states = torch.from_numpy(sample_experiences[0]).float().to(self.device)
			sample_actions = torch.from_numpy(sample_experiences[1]).float().to(self.device)
			sample_qvalues = option.solver.get_qvalues(sample_states, sample_actions)
			return sample_qvalues.mean().item()
		return 0.0

	@staticmethod
	def get_next_state_from_experiences(experiences):
		return experiences[-1][-1]

	@staticmethod
	def get_reward_from_experiences(experiences):
		total_reward = 0.
		for experience in experiences:
			reward = experience[2]
			total_reward += reward
		return total_reward

	def should_create_more_options(self):
		""" Continue chaining as long as any chain is still accepting new options. """
		return any([chain.should_continue_chaining(self.chains) for chain in self.chains])

	def get_intersecting_options(self):
		""" Iterate through all the skill chains and return all pairs of options that have intersecting initiation sets. """
		intersecting_pairs = []
		for chain in self.chains:  # type: SkillChain
			intersecting_options = chain.detect_intersection_with_other_chains(self.chains)
			if intersecting_options is not None:

				# Only care about intersections between forward chains
				if not intersecting_options[0].backward_option and not intersecting_options[1].backward_option:
					intersecting_pairs.append(intersecting_options)

		return intersecting_pairs

	def get_current_salient_events(self):
		""" Get a list of states that satisfy final target events (i.e not init of an option) in the MDP so far. """
		return [self.mdp.goal_position, self.mdp.key_position]

	def get_original_salient_events(self):
		original_salient_events = self.mdp.get_original_target_events()
		if self.start_state_salience:
			original_salient_events.append(self.mdp.is_start_state)
		return original_salient_events

	def get_backward_chain_start_states(self, option1, option2):
		"""
		Given 2 forward options, return the target states associated with the corresponding
		forward skill-chains. These target states serve as init states for the new
		backward chain.
		Args:
			option1 (Option)
			option2 (Option)

		Returns:
			states (list): List of states to chain back to in the backward chain
		"""
		chain1 = self.chains[option1.chain_id - 1]  # Forward chain 1
		chain2 = self.chains[option2.chain_id - 1]  # Forward chain 2
		assert not chain1.is_backward_chain
		assert not chain2.is_backward_chain
		start1 = chain1.target_position
		start2 = chain2.target_position
		return start1, start2

	def create_intersection_salient_event(self, option1, option2):
		"""
		When 2 chains intersect, we treat the region of intersection as a target region. This method
		returns an Option whose termination region is such a region of intersection.
		Args:
			option1 (Option): intersecting option from first chain
			option2 (Option): intersecting option from second chain

		Returns:
			new_option (Option): whose termination set is the intersection of initiation sets of o1 and o2
		"""
		intersecting_chain_idx = (option1.chain_id, option2.chain_id)
		chain1 = self.chains[option1.chain_id - 1]
		chain2 = self.chains[option2.chain_id - 1]

		assert not chain1.is_backward_chain, "Intersection should only be considered between forward chains"
		assert not chain2.is_backward_chain, "Intersection should only be considered between forward chains"
		assert option1.chain_id != option2.chain_id, f"Chain idx: {option1.chain_id, option2.chain_id}"

		# Create a new intersection salient event for chain1 & chain2 only if we haven't done so already
		if intersecting_chain_idx not in self.current_intersecting_chains and \
				tuple(reversed(intersecting_chain_idx)) not in self.current_intersecting_chains:
			self.current_intersecting_chains.append(intersecting_chain_idx)

			# TODO: Use the goals of the intersecting forward chains as initiations for the backward chain
			# The forward intersecting chains were targeting two different salient events
			# Those two salient events are going to serve as the initiation condition of the options
			# in the new chain we are about to create (during gestation only)
			salient1 = chain1.target_predicate
			salient2 = chain2.target_predicate

			new_option = Option(self.mdp, name=f"chain_{intersecting_chain_idx}_intersection_option",
								  global_solver=self.global_option.solver,
								  lr_actor=option1.solver.actor_learning_rate,
								  lr_critic=option1.solver.critic_learning_rate,
								  ddpg_batch_size=option1.solver.batch_size,
								  subgoal_reward=self.subgoal_reward,
								  buffer_length=self.buffer_length,
								  classifier_type=self.classifier_type,
								  num_subgoal_hits_required=self.num_subgoal_hits_required,
								  seed=self.seed, parent=None,  max_steps=self.max_steps,
								  enable_timeout=self.enable_option_timeout, chain_id=len(self.chains)+1,
								  intersecting_options=[option1, option2],
								  initialize_everywhere=True, max_num_children=2,
								  writer=self.writer, device=self.device, dense_reward=self.dense_reward,
								  is_backward_option=True)

			# 1. The new backward chain will chain backward to the salient events corresponding to the intersecting forward chains
			# 2. The target_predicate for the backward chain is set inside the SkillChain class - it is the region of
			#    intersection between the two intersecting chains
			new_chain_start_states = self.get_backward_chain_start_states(option1, option2)
			new_chain = SkillChain(start_states=new_chain_start_states, target_salient_event=None, chain_id=len(self.chains)+1,
								   options=[], intersecting_options=[option1, option2],
								   mdp_start_states=self.s0, is_backward_chain=True)
			self.chains.append(new_chain)

			# Allow agent to use this option to encourage chaining to it
			self._augment_agent_with_new_option(new_option, 0.)
			new_option.initialize_with_global_ddpg()

			# Also reset the exploration module to encourage adding skills to the new chain
			# self.agent_over_options.density_model.reset()
			self.initialize_policy_over_options()

			return new_option

		return None

	def manage_intersecting_skill_chains(self):
		# Detect intersection salience to start building skill graph
		intersecting_options = self.get_intersecting_options()

		if intersecting_options:
			for intersecting_1, intersecting_2 in intersecting_options:
				new_option = self.create_intersection_salient_event(intersecting_1, intersecting_2)
				if new_option is not None:
					self.untrained_options.append(new_option)

	def manage_skill_chain_to_start_state(self, option):
		"""
		When we complete creating a skill-chain targeting a salient event,
		we want to create a new skill chain that goes in the opposite direction.
		Args:
			option (Option)

		"""
		chain = self.chains[option.chain_id - 1]  # type: SkillChain
		assert not chain.is_backward_chain, "Only applicable for forward chains"
		if chain.chained_till_start_state() and not chain.has_backward_chain:
			chain.has_backward_chain = True

			# 1. The new skill chain will chain back until it covers `start_states`
			# 2. The options in this new backward chain will use `start_predicate` as the
			#	 default initiation set during gestation
			start_states = [chain.target_salient_event.target_state]
			init_salient_event = chain.target_salient_event

			# These target predicates are used to determine the goal of the new skill chain
			target_salient_event = self.mdp.get_start_state_salient_event()

			new_chain = SkillChain(start_states=start_states, mdp_start_states=self.s0,
								   target_salient_event=target_salient_event,
								   options=[], chain_id=len(self.chains)+1,
								   intersecting_options=[], is_backward_chain=True)
			self.chains.append(new_chain)

			# Create a new option and equip the SkillChainingAgent with it
			new_option = Option(self.mdp, name=f"chain_{chain.chain_id}_backward_option",
								global_solver=self.global_option.solver,
								lr_actor=self.global_option.solver.actor_learning_rate,
								lr_critic=self.global_option.solver.critic_learning_rate,
								ddpg_batch_size=self.global_option.solver.batch_size,
								subgoal_reward=self.subgoal_reward,
								buffer_length=self.buffer_length,
								classifier_type=self.classifier_type,
								num_subgoal_hits_required=self.num_subgoal_hits_required,
								seed=self.seed, parent=None, max_steps=self.max_steps,
								enable_timeout=self.enable_option_timeout, chain_id=len(self.chains),
								intersecting_options=[],
								initialize_everywhere=True, max_num_children=1,
								writer=self.writer, device=self.device, dense_reward=self.dense_reward,
								is_backward_option=True,
								init_salient_event=init_salient_event,
								target_salient_event=target_salient_event)

			self._augment_agent_with_new_option(new_option, 0.)
			new_option.initialize_with_global_ddpg()
			self.untrained_options.append(new_option)

	def skill_chaining(self, num_episodes, num_steps):

		# For logging purposes
		per_episode_scores = []
		per_episode_durations = []
		last_10_scores = deque(maxlen=10)
		last_10_durations = deque(maxlen=10)

		for episode in range(num_episodes):

			self.mdp.reset()
			score = 0.
			step_number = 0

			# TODO: Does this flag's logic need to be changed? YES when there is a list of target events in the MDP
			uo_episode_terminated = False

			state = deepcopy(self.mdp.cur_state)
			self.init_states.append(deepcopy(state))
			experience_buffer = []
			state_buffer = []
			episode_option_executions = defaultdict(lambda : 0)

			while step_number < num_steps:
				# pdb.set_trace()
				experiences, reward, state, steps = self.take_action(state, episode, step_number)
				score += reward
				step_number += steps
				for experience in experiences:
					experience_buffer.append(experience)
					state_buffer.append(experience[0])

				# Don't forget to add the last s' to the buffer
				if state.is_terminal() or (step_number == num_steps - 1):
					state_buffer.append(state)

				# In the skill-tree setting we could be learning many options at the current time
				# We must iterate through all such options and check if the current transition
				# triggered the termination condition of any such option
				for untrained_option in self.untrained_options:

					if untrained_option.is_term_true(state) and (not uo_episode_terminated) and \
							self.max_num_options > 0 and untrained_option.get_training_phase() == "gestation" and \
							untrained_option.is_valid_init_data(state_buffer) and not untrained_option.initialize_everywhere:

						print("\rTriggered termination condition of ", untrained_option)
						uo_episode_terminated = True
						if untrained_option.train(experience_buffer, state_buffer, episode):

							# If the option can be initiated from anywhere, it has already been added to
							# the list of actions available to the agent (so that it may be executed before
							# we train its one-class classifier)
							if not untrained_option.initialize_everywhere:
								self._augment_agent_with_new_option(untrained_option, self.init_q)

							# Visualize option you just learned
							if untrained_option.classifier_type == "ocsvm":
								plot_one_class_initiation_classifier(untrained_option, episode, args.experiment_name)
							else:
								plot_two_class_classifier(untrained_option, episode, args.experiment_name)

					# In the skill-graph setting, we have to check if the current option's chain
					# is still accepting new options
					if self.should_create_more_options() and untrained_option.get_training_phase() == "initiation_done" \
							and self.chains[untrained_option.chain_id - 1].should_continue_chaining(self.chains):

						# Debug visualization
						plot_two_class_classifier(untrained_option, episode, args.experiment_name)

						# We fix the learned option's initiation set and remove it from the list of target events
						self.untrained_options.remove(untrained_option)

						new_options = self.create_children_options(untrained_option)

						# Iterate through all the child options and add them to the skill tree 1 by 1
						for new_option in new_options:  # type: Option
							if new_option is not None:
								self.untrained_options.append(new_option)
								self.add_negative_examples(new_option)

								# If initialize_everywhere is True, also add to trained_options
								if new_option.initialize_everywhere:
									self._augment_agent_with_new_option(new_option, self.init_q)
									new_option.initialize_with_global_ddpg()

					# If the current option's initiation set covers one of the target salient events
					# (and the current is in a backward chain), then remove that salient event from the
					# list of target events. This way, we will generate options that start from other
					# salient events and end up in the corresponding intersection salient event
					if untrained_option.get_training_phase() == "initiation_done" and untrained_option.backward_option:
						self.mdp.satisfy_target_event(untrained_option)

					if untrained_option.get_training_phase() == "initiation_done" and \
							not untrained_option.backward_option and self.start_state_salience:
						self.manage_skill_chain_to_start_state(untrained_option)

				if not self.start_state_salience:
					self.manage_intersecting_skill_chains()

				if state.is_terminal():
					break

			last_10_scores.append(score)
			last_10_durations.append(step_number)
			per_episode_scores.append(score)
			per_episode_durations.append(step_number)

			self._log_dqn_status(episode, last_10_scores, episode_option_executions, last_10_durations)

		return per_episode_scores, per_episode_durations

	def _log_dqn_status(self, episode, last_10_scores, episode_option_executions, last_10_durations):

		print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tGO Eps: {:.2f}'.format(
			episode, np.mean(last_10_scores), np.mean(last_10_durations), self.global_option.solver.epsilon))

		self.num_options_history.append(len(self.trained_options))

		if self.writer is not None:
			self.writer.add_scalar("Episodic scores", last_10_scores[-1], episode)

		if episode % 10 == 0:
			print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tGO Eps: {:.2f}'.format(
				episode, np.mean(last_10_scores), np.mean(last_10_durations), self.global_option.solver.epsilon))

		if episode > 0 and episode % 100 == 0:
			# eval_score, trajectory = self.trained_forward_pass(render=False)
			eval_score, trajectory = 0., []

			self.validation_scores.append(eval_score)
			print("\rEpisode {}\tValidation Score: {:.2f}".format(episode, eval_score))

		if self.generate_plots and episode % 10 == 0 and episode > 0:

			visualize_best_option_to_take(self.agent_over_options, episode, self.seed, args.experiment_name)

			for option in self.trained_options:
				visualize_ddpg_replay_buffer(option.solver, episode, self.seed, args.experiment_name)
				visualize_ddpg_shaped_rewards(self.global_option, option, episode, self.seed, args.experiment_name)
				visualize_dqn_shaped_rewards(self.agent_over_options, option, episode, self.seed, args.experiment_name)

		for trained_option in self.trained_options:  # type: Option
			self.num_option_executions[trained_option.name].append(episode_option_executions[trained_option.name])
			if self.writer is not None:
				self.writer.add_scalar("{}_executions".format(trained_option.name),
									   episode_option_executions[trained_option.name], episode)

	def save_all_models(self):
		for option in self.trained_options: # type: Option
			save_model(option.solver, args.episodes, best=False)

	def save_all_scores(self, pretrained, scores, durations):
		print("\rSaving training and validation scores..")
		training_scores_file_name = "sc_pretrained_{}_training_scores_{}.pkl".format(pretrained, self.seed)
		training_durations_file_name = "sc_pretrained_{}_training_durations_{}.pkl".format(pretrained, self.seed)
		validation_scores_file_name = "sc_pretrained_{}_validation_scores_{}.pkl".format(pretrained, self.seed)
		num_option_history_file_name = "sc_pretrained_{}_num_options_per_epsiode_{}.pkl".format(pretrained, self.seed)

		if self.log_dir:
			training_scores_file_name = os.path.join(self.log_dir, training_scores_file_name)
			training_durations_file_name = os.path.join(self.log_dir, training_durations_file_name)
			validation_scores_file_name = os.path.join(self.log_dir, validation_scores_file_name)
			num_option_history_file_name = os.path.join(self.log_dir, num_option_history_file_name)

		with open(training_scores_file_name, "wb+") as _f:
			pickle.dump(scores, _f)
		with open(training_durations_file_name, "wb+") as _f:
			pickle.dump(durations, _f)
		with open(validation_scores_file_name, "wb+") as _f:
			pickle.dump(self.validation_scores, _f)
		with open(num_option_history_file_name, "wb+") as _f:
			pickle.dump(self.num_options_history, _f)

	def perform_experiments(self):
		for option in self.trained_options:
			visualize_dqn_replay_buffer(option.solver, args.experiment_name)

		for i, o in enumerate(self.trained_options):
			plt.subplot(1, len(self.trained_options), i + 1)
			plt.plot(self.option_qvalues[o.name])
			plt.title(o.name)
		plt.savefig("value_function_plots/{}/sampled_q_so_{}.png".format(args.experiment_name, self.seed))
		plt.close()

		for option in self.trained_options:
			visualize_next_state_reward_heat_map(option.solver, args.episodes, args.experiment_name)

		for i, o in enumerate(self.trained_options):
			plt.subplot(1, len(self.trained_options), i + 1)
			plt.plot(o.taken_or_not)
			plt.title(o.name)
		plt.savefig("value_function_plots/{}_taken_or_not_{}.png".format(args.experiment_name, self.seed))
		plt.close()

	def trained_forward_pass(self, render=True):
		"""
		Called when skill chaining has finished training: execute options when possible and then atomic actions
		Returns:
			overall_reward (float): score accumulated over the course of the episode.
		"""
		self.mdp.reset()
		state = deepcopy(self.mdp.init_state)
		overall_reward = 0.
		self.mdp.render = render
		num_steps = 0
		option_trajectories = []

		while not state.is_terminal() and num_steps < self.max_steps:
			selected_option = self.act(state)

			option_reward, next_state, num_steps, option_state_trajectory = selected_option.trained_option_execution(self.mdp, num_steps)
			overall_reward += option_reward

			# option_state_trajectory is a list of (o, s) tuples
			option_trajectories.append(option_state_trajectory)

			state = next_state

		return overall_reward, option_trajectories

	def create_test_time_goal_option(self, train_goal_option):
		pass

	def should_planning_run_loop_terminate(self, state, goal_state):
		"""
		There are 3 possible cases:
		1. When the goal_state is a salient event we already know how to get to:
		   In this case, we only want to terminate when we reach the goal_state
		2. When the goal_state is not a salient event we know how to get to, but is inside the initiation
		   set of some option that we have already learned:
		   In this case, we want to terminate when we reach the option that contains that goal_state, but
		   not execute the option itself. Later, we will learn a new option that will drive us to that goal state.
		3. When the goal_state is not a salient event we know about AND it is not inside the initiation set of
		   any of the options that we have learned:
		   In this case, we have to find the option whose initiation set is "closest" to the goal state.
		   We can either use a pre-specified measure (eg euclidean distance) or use the space of option value functions
		   to find the "closest" option.
		Args:
			state (State)
			goal_state (State)

		Returns:
			should_terminate (bool)
		"""

		# TODO: Maybe get the salient events from self.chains?
		# Case 1: goal_state satisfies one of the salient events we already know about
		salient_events = self.get_original_salient_events()
		satisfied_salients = [salient_event for salient_event in salient_events if salient_event(goal_state)]
		if len(satisfied_salients) > 0:
			return satisfied_salients[0](state)

		chains_with_goal_state = [chain for chain in self.chains if chain.state_in_chain(goal_state)]
		options_with_goal_state = [chain.get_option_for_state(goal_state) for chain in chains_with_goal_state]
		if len(options_with_goal_state) > 0:  # TODO: How to pick an option from this list of feasible choices?
			target_option = options_with_goal_state[0]
			return target_option.is_init_true(state)

		raise NotImplementedError("Not implemented case 3")

	def planning_run_loop(self, plan_graph, goal_state, start_state=None):
		"""
		This is a test-time run loop that uses planning to select options when it can.
		Args:
			plan_graph (GraphSearch)
			goal_state (State)
			start_state (State)

		Returns:
			accumulated_reward (float)
			executed_options (list)
			reached_goal (bool)
		"""

		def _select_option(s, sg):
			""" Given a start state `s` and a goal state `sg` pick an option to execute.
				Use planning if possible, else revert to the trained policy over options. """
			if plan_graph.does_path_exist(s, sg):
				plan = plan_graph.get_path_to_execute(s, sg)
				if len(plan) > 0:
					admissible_options = [o for o in plan if o.is_init_true(s) and not o.is_term_true(s)]
					if len(admissible_options) > 0:
						return admissible_options[0]
			return self.act(s)

		self.mdp.reset()

		if start_state is not None:
			start_position = start_state.position if isinstance(start_state, State) else start_state[:2]
			self.mdp.set_xy(start_position)

		state = deepcopy(self.mdp.cur_state)
		overall_reward = 0.
		num_steps = 0
		option_trajectories = []
		should_terminate_run_loop = False
		goal_position = goal_state.position if isinstance(goal_state, State) else goal_state

		while not should_terminate_run_loop and num_steps < self.max_steps:

			selected_option = _select_option(state, goal_state)

			option_reward, next_state, num_steps, option_state_trajectory = selected_option.trained_option_execution(self.mdp, num_steps)
			overall_reward += option_reward

			# option_state_trajectory is a list of (o, s) tuples
			option_trajectories.append(option_state_trajectory)

			state = next_state

			should_terminate_run_loop = self.should_planning_run_loop_terminate(state, goal_state)
			reached_goal_state = np.linalg.norm(state.position - goal_position) <= 0.6

			if should_terminate_run_loop and not reached_goal_state:  # Case II
				# Create new option
				pass

		return overall_reward, option_trajectories

	def perform_test_rollouts(self, num_rollouts):
		""" Perform test rollouts to collect statistics on option success rates.
		    We can then use those success rates to set edge weights in our
		    abstract skill graph.
		"""
		summed_reward = 0.
		for _ in tqdm(range(num_rollouts), desc="Performing DSC Test Rollouts"):
			r, _ = self.trained_forward_pass(render=False)
			summed_reward += r
		return summed_reward

	def construct_plan_graph(self):

		# Perform test rollouts to determine edge weights in the plan graph
		self.perform_test_rollouts(num_rollouts=1000)

		# Construct the plan graph
		plan_graph = GraphSearch()
		plan_graph.construct_graph(self.chains)

		# Visualize the graph
		file_name = f"value_function_plots/{args.experiment_name}/plan_graph_seed_{self.seed}.png"
		plan_graph.visualize_plan_graph(file_name)

		return plan_graph

	def measure_success(self, goal_state, start_state=None):
		successes = 0
		for _ in range(100):
			self.planning_run_loop(graph, goal_state, start_state)
			goal_pos = goal_state
			is_terminal = np.linalg.norm(overall_mdp.cur_state.position-goal_pos) <= 0.6
			if is_terminal: successes += 1
		return successes


def create_log_dir(experiment_name):
	path = os.path.join(os.getcwd(), experiment_name)
	try:
		os.mkdir(path)
	except OSError:
		print("Creation of the directory %s failed" % path)
	else:
		print("Successfully created the directory %s " % path)
	return path


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--experiment_name", type=str, help="Experiment Name")
	parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
	parser.add_argument("--env", type=str, help="name of gym environment", default="Pendulum-v0")
	parser.add_argument("--pretrained", type=bool, help="whether or not to load pretrained options", default=False)
	parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
	parser.add_argument("--episodes", type=int, help="# episodes", default=200)
	parser.add_argument("--steps", type=int, help="# steps", default=1000)
	parser.add_argument("--subgoal_reward", type=float, help="SkillChaining subgoal reward", default=0.)
	parser.add_argument("--lr_a", type=float, help="DDPG Actor learning rate", default=1e-4)
	parser.add_argument("--lr_c", type=float, help="DDPG Critic learning rate", default=1e-3)
	parser.add_argument("--ddpg_batch_size", type=int, help="DDPG Batch Size", default=64)
	parser.add_argument("--render", type=bool, help="Render the mdp env", default=False)
	parser.add_argument("--option_timeout", type=bool, help="Whether option times out at 200 steps", default=False)
	parser.add_argument("--generate_plots", type=bool, help="Whether or not to generate plots", default=False)
	parser.add_argument("--tensor_log", type=bool, help="Enable tensorboard logging", default=False)
	parser.add_argument("--control_cost", type=bool, help="Penalize high actuation solutions", default=False)
	parser.add_argument("--dense_reward", type=bool, help="Use dense/sparse rewards", default=False)
	parser.add_argument("--max_num_options", type=int, help="Max number of options we can learn", default=5)
	parser.add_argument("--num_subgoal_hits", type=int, help="Number of subgoal hits to learn an option", default=3)
	parser.add_argument("--buffer_len", type=int, help="buffer size used by option to create init sets", default=20)
	parser.add_argument("--classifier_type", type=str, help="ocsvm/elliptic for option initiation clf", default="ocsvm")
	parser.add_argument("--init_q", type=str, help="compute/zero", default="zero")
	parser.add_argument("--use_smdp_update", type=bool, help="sparse/SMDP update for option policy", default=False)
	parser.add_argument("--use_start_state_salience", action="store_true", default=False)
	args = parser.parse_args()

	if args.env == "point-reacher":
		from simple_rl.tasks.point_reacher.PointReacherMDPClass import PointReacherMDP
		overall_mdp = PointReacherMDP(seed=args.seed, dense_reward=args.dense_reward, render=args.render)
		state_dim = 6
		action_dim = 2
	elif "reacher" in args.env.lower():
		from simple_rl.tasks.dm_fixed_reacher.FixedReacherMDPClass import FixedReacherMDP
		overall_mdp = FixedReacherMDP(seed=args.seed, difficulty=args.difficulty, render=args.render)
		state_dim = overall_mdp.init_state.features().shape[0]
		action_dim = overall_mdp.env.action_spec().minimum.shape[0]
	elif "maze" in args.env.lower():
		from simple_rl.tasks.point_maze.PointMazeMDPClass import PointMazeMDP
		overall_mdp = PointMazeMDP(dense_reward=args.dense_reward, seed=args.seed, render=args.render)
		state_dim = 6
		action_dim = 2
	elif "point" in args.env.lower():
		from simple_rl.tasks.point_env.PointEnvMDPClass import PointEnvMDP
		overall_mdp = PointEnvMDP(control_cost=args.control_cost, render=args.render)
		state_dim = 4
		action_dim = 2
	else:
		from simple_rl.tasks.gym.GymMDPClass import GymMDP
		overall_mdp = GymMDP(args.env, render=args.render)
		state_dim = overall_mdp.env.observation_space.shape[0]
		action_dim = overall_mdp.env.action_space.shape[0]
		overall_mdp.env.seed(args.seed)

	# Create folders for saving various things
	logdir = create_log_dir(args.experiment_name)
	create_log_dir("saved_runs")
	create_log_dir("value_function_plots")
	create_log_dir("initiation_set_plots")
	create_log_dir("value_function_plots/{}".format(args.experiment_name))
	create_log_dir("initiation_set_plots/{}".format(args.experiment_name))

	print("Training skill chaining agent from scratch with a subgoal reward {}".format(args.subgoal_reward))
	print("MDP InitState = ", overall_mdp.init_state)

	q0 = 0. if args.init_q == "zero" else None

	chainer = SkillChaining(overall_mdp, args.steps, args.lr_a, args.lr_c, args.ddpg_batch_size,
							seed=args.seed, subgoal_reward=args.subgoal_reward,
							log_dir=logdir, num_subgoal_hits_required=args.num_subgoal_hits,
							enable_option_timeout=args.option_timeout, init_q=q0, use_full_smdp_update=args.use_smdp_update,
							generate_plots=args.generate_plots, tensor_log=args.tensor_log, device=args.device,
							start_state_salience=args.use_start_state_salience)
	episodic_scores, episodic_durations = chainer.skill_chaining(args.episodes, args.steps)

	# Log performance metrics
	# chainer.save_all_models()
	# chainer.perform_experiments()
	chainer.save_all_scores(args.pretrained, episodic_scores, episodic_durations)

	graph = chainer.construct_plan_graph()
