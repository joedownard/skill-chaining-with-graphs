import numpy as np
from scipy.spatial import distance
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.mdp import MDP, State


class GoalDirectedMDP(MDP):
    def __init__(self, actions, transition_func, reward_func, init_state,
                 salient_positions, task_agnostic, goal_state=None, goal_tolerance=0.6):

        self.salient_positions = salient_positions
        self.task_agnostic = task_agnostic
        self.goal_tolerance = goal_tolerance
        self.goal_state = goal_state
        self.dense_reward = False

        if not task_agnostic:
            assert self.goal_state is not None, self.goal_state

        self._initialize_salient_events()

        MDP.__init__(self, actions, transition_func, reward_func, init_state)

    def _initialize_salient_events(self):
        # Set the current target events in the MDP
        self.current_salient_events = [SalientEvent(pos, event_idx=i + 1) for i, pos in
                                       enumerate(self.salient_positions)]

        # Set an ever expanding list of salient events - we need to keep this around to call is_term_true on trained options
        self.original_salient_events = [event for event in self.current_salient_events]

        # In some MDPs, we use a predicate to determine if we are at the start state of the MDP
        self.start_state_salient_event = SalientEvent(target_state=self.init_state.position, event_idx=0)

        # Keep track of all the salient events ever created in this MDP
        self.all_salient_events_ever = [event for event in self.current_salient_events]

        # Make sure that we didn't create multiple copies of the same events
        self._ensure_all_events_are_the_same()

    def _ensure_all_events_are_the_same(self):
        for e1, e2, e3 in zip(self.current_salient_events, self.original_salient_events, self.all_salient_events_ever):
            assert id(e1) == id(e2) == id(e3)

    def get_current_target_events(self):
        """ Return list of predicate functions that indicate salience in this MDP. """
        return self.current_salient_events

    def get_original_target_events(self):
        return self.original_salient_events

    def get_all_target_events_ever(self):
        return self.all_salient_events_ever

    def add_new_target_event(self, new_event):
        if new_event not in self.current_salient_events:
            self.current_salient_events.append(new_event)

        if new_event not in self.all_salient_events_ever:
            self.all_salient_events_ever.append(new_event)

    def is_start_state(self, state):
        pos = self._get_position(state)
        s0 = self.init_state.position
        return np.linalg.norm(pos - s0) <= self.goal_tolerance

    def batched_is_start_state(self, position_matrix):
        s0 = self.init_state.position
        in_start_pos = distance.cdist(position_matrix, s0[None, :]) <= self.goal_tolerance
        return in_start_pos.squeeze(1)

    def get_start_state_salient_event(self):
        return self.start_state_salient_event

    def satisfy_target_event(self, chains):
        """
        Once a salient event has both forward and backward options related to it,
        we no longer need to maintain it as a target_event. This function will find
        the salient event that corresponds to the input state and will remove that
        event from the list of target_events. Additionally, we have to ensure that
        the chain corresponding to `option` is "completed".

        A target event is satisfied when all chains targeting it are completed.

        Args:
            chains (list)

        Returns:

        """
        for salient_event in self.get_current_target_events():  # type: SalientEvent
            satisfied_salience = self.should_remove_salient_event_from_mdp(salient_event, chains)
            if satisfied_salience and (salient_event in self.current_salient_events):
                self.current_salient_events.remove(salient_event)

    @staticmethod
    def should_remove_salient_event_from_mdp(salient_event, chains):
        incoming_chains = [chain for chain in chains if chain.target_salient_event == salient_event]
        outgoing_chains = [chain for chain in chains if chain.init_salient_event == salient_event]

        event_chains = incoming_chains + outgoing_chains
        event_chains_completed = all([chain.is_chain_completed(chains) for chain in event_chains])
        satisfied = len(incoming_chains) > 0 and len(outgoing_chains) and event_chains_completed

        return satisfied

    def is_goal_state(self, state):
        if self.task_agnostic:
            return False
        raise NotImplementedError(self.task_agnostic)

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(GoalDirectedMDP, self).execute_agent_action(action)
        return reward, next_state

    @staticmethod
    def _get_position(state):
        position = state.position if isinstance(state, State) else state[:2]
        return position