#!/usr/bin/env python3
"""Q-Learner Bot wrapper for OpenSpiel.

This module provides a Bot wrapper around trained Q-learning agents,
enabling them to be used in interactive play and bot vs bot evaluation.
"""

import numpy as np
import pyspiel
from open_spiel.python.algorithms import tabular_qlearner

class QLearnerBot(pyspiel.Bot):
    """A bot that uses a trained Q-learning agent for action selection.

    This bot wraps a trained Q-learner and makes it compatible with the
    OpenSpiel Bot interface for interactive play.
    """

    def __init__(self, qlearner_agent, use_greedy=True):
        """Initialize the Q-learner bot.

        Args:
            qlearner_agent: A trained QLearner instance
            use_greedy: If True, always take the greedy action. If False, use epsilon-greedy.
        """
        pyspiel.Bot.__init__(self)
        self._agent = qlearner_agent
        self._use_greedy = use_greedy
        self._use_info_state = True  # Flag to avoid repeated exception handling

    def player_id(self):
        """Return the player ID this bot represents."""
        return self._agent._player_id

    def restart_at(self, state):
        """Called when the bot should restart from the given state."""
        pass

    def step(self, state):
        """Choose an action for the given state.

        Args:
            state: The current game state

        Returns:
            The action to take
        """
        # Get legal actions - handle simultaneous vs sequential games
        if state.is_simultaneous_node():
            # For simultaneous games, get player-specific actions
            legal_actions = state.legal_actions(self._agent._player_id)
        else:
            # For sequential games, get all legal actions
            legal_actions = state.legal_actions()

        if not legal_actions:
            return pyspiel.INVALID_ACTION

        # Get state representation for Q-learning
        if self._use_info_state:
            try:
                info_state = state.information_state_string(self._agent._player_id)
            except:
                # Information state not available, disable for future calls
                self._use_info_state = False
                if state.is_simultaneous_node():
                    info_state = f"P{self._agent._player_id}:{str(state)}"
                else:
                    info_state = str(state)
        else:
            # Use regular state string (info state not available)
            if state.is_simultaneous_node():
                info_state = f"P{self._agent._player_id}:{str(state)}"
            else:
                info_state = str(state)

        if self._use_greedy:
            # Always take greedy action
            epsilon = 0.0
        else:
            # Use current epsilon value
            epsilon = self._agent._epsilon

        # Use the Q-learner's epsilon-greedy policy
        action, _ = self._agent._epsilon_greedy(info_state, legal_actions, epsilon)

        return action

class QLearnerPolicy:
    """A policy interface for Q-learner that can be used with PolicyBot."""

    def __init__(self, qlearner_agent, use_greedy=True):
        """Initialize the Q-learner policy.

        Args:
            qlearner_agent: A trained QLearner instance
            use_greedy: If True, return greedy probabilities. If False, use epsilon-greedy.
        """
        self._agent = qlearner_agent
        self._use_greedy = use_greedy
        self._use_info_state = True  # Flag to avoid repeated exception handling

    def action_probabilities(self, state, player_id):
        """Return action probabilities for the given state.

        Args:
            state: The current game state
            player_id: The player to get probabilities for

        Returns:
            Dictionary mapping actions to probabilities
        """
        if player_id != self._agent._player_id:
            # Not our player, return uniform random
            legal_actions = state.legal_actions(player_id)
            if not legal_actions:
                return {}
            prob = 1.0 / len(legal_actions)
            return {action: prob for action in legal_actions}

        # Get legal actions - handle simultaneous vs sequential games
        if state.is_simultaneous_node():
            # For simultaneous games, get player-specific actions
            legal_actions = state.legal_actions(player_id)
        else:
            # For sequential games, get all legal actions
            legal_actions = state.legal_actions()

        if not legal_actions:
            return {}

        # Get state representation for Q-learning
        if self._use_info_state:
            try:
                info_state = state.information_state_string(player_id)
            except:
                # Information state not available, disable for future calls
                self._use_info_state = False
                if state.is_simultaneous_node():
                    info_state = f"P{player_id}:{str(state)}"
                else:
                    info_state = str(state)
        else:
            # Use regular state string (info state not available)
            if state.is_simultaneous_node():
                info_state = f"P{player_id}:{str(state)}"
            else:
                info_state = str(state)

        if self._use_greedy:
            epsilon = 0.0
        else:
            epsilon = self._agent._epsilon

        # Get action probabilities from Q-learner
        _, probs = self._agent._epsilon_greedy(info_state, legal_actions, epsilon)

        # Convert to dictionary format
        action_probs = {}
        for i, action in enumerate(range(self._agent._num_actions)):
            if action in legal_actions and probs[action] > 0:
                action_probs[action] = probs[action]

        return action_probs

def create_qlearner_bot(qlearner_agent, use_greedy=True):
    """Factory function to create a Q-learner bot.

    Args:
        qlearner_agent: A trained QLearner instance
        use_greedy: If True, bot will always take greedy actions

    Returns:
        A QLearnerBot instance
    """
    return QLearnerBot(qlearner_agent, use_greedy)