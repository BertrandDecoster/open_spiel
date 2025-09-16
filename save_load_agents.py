#!/usr/bin/env python3
"""Utilities for saving and loading trained Q-learning agents.

This module provides functions to persist and restore Q-learning agents,
enabling reuse of trained models for evaluation and interactive play.
"""

import pickle
import os
from typing import Dict, Any
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python import rl_tools

def save_qlearner(agent: tabular_qlearner.QLearner, filepath: str) -> None:
    """Save a Q-learning agent to disk.

    Args:
        agent: The trained QLearner instance to save
        filepath: Path where to save the agent (should end with .pkl)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Prepare data to save
    agent_data = {
        'q_values': dict(agent._q_values),  # Convert defaultdict to regular dict
        'player_id': agent._player_id,
        'num_actions': agent._num_actions,
        'step_size': agent._step_size,
        'discount_factor': agent._discount_factor,
        'centralized': agent._centralized,
        'epsilon': agent._epsilon,
        'agent_type': 'QLearner'
    }

    # Save epsilon schedule state if available
    if hasattr(agent._epsilon_schedule, 'value'):
        agent_data['epsilon_schedule_value'] = agent._epsilon_schedule.value
    if hasattr(agent._epsilon_schedule, '_num_steps'):
        agent_data['epsilon_schedule_steps'] = agent._epsilon_schedule._num_steps

    with open(filepath, 'wb') as f:
        pickle.dump(agent_data, f)

    print(f"Saved Q-learner agent to {filepath}")

def load_qlearner(filepath: str, num_actions: int = None) -> tabular_qlearner.QLearner:
    """Load a Q-learning agent from disk.

    Args:
        filepath: Path to the saved agent file
        num_actions: Number of actions (will use saved value if not provided)

    Returns:
        A QLearner instance with loaded Q-values
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Agent file not found: {filepath}")

    with open(filepath, 'rb') as f:
        agent_data = pickle.load(f)

    # Validate agent type
    if agent_data.get('agent_type') != 'QLearner':
        raise ValueError(f"Invalid agent type: {agent_data.get('agent_type')}")

    # Use provided num_actions or saved value
    if num_actions is None:
        num_actions = agent_data['num_actions']

    # Create epsilon schedule (use constant schedule with saved epsilon)
    epsilon_schedule = rl_tools.ConstantSchedule(agent_data.get('epsilon', 0.01))

    # Create new agent with same parameters
    agent = tabular_qlearner.QLearner(
        player_id=agent_data['player_id'],
        num_actions=num_actions,
        step_size=agent_data.get('step_size', 0.1),
        epsilon_schedule=epsilon_schedule,
        discount_factor=agent_data.get('discount_factor', 0.99),
        centralized=agent_data.get('centralized', False)
    )

    # Restore Q-values
    agent._q_values.clear()
    for info_state, action_values in agent_data['q_values'].items():
        agent._q_values[info_state] = action_values

    # Restore epsilon
    agent._epsilon = agent_data.get('epsilon', 0.01)

    print(f"Loaded Q-learner agent from {filepath}")
    print(f"  Player ID: {agent._player_id}")
    print(f"  Q-table size: {len(agent._q_values)} states")
    print(f"  Current epsilon: {agent._epsilon}")

    return agent

def save_multiple_agents(agents: list, base_path: str, game_name: str) -> list:
    """Save multiple agents with automatic naming.

    Args:
        agents: List of QLearner instances
        base_path: Base directory to save agents
        game_name: Name of the game for file naming

    Returns:
        List of filepaths where agents were saved
    """
    filepaths = []
    for i, agent in enumerate(agents):
        filename = f"{game_name}_player{agent._player_id}_qlearner.pkl"
        filepath = os.path.join(base_path, filename)
        save_qlearner(agent, filepath)
        filepaths.append(filepath)

    return filepaths

def load_multiple_agents(filepaths: list) -> list:
    """Load multiple agents from a list of filepaths.

    Args:
        filepaths: List of paths to saved agent files

    Returns:
        List of loaded QLearner instances
    """
    agents = []
    for filepath in filepaths:
        agent = load_qlearner(filepath)
        agents.append(agent)

    return agents

def get_agent_info(filepath: str) -> Dict[str, Any]:
    """Get information about a saved agent without fully loading it.

    Args:
        filepath: Path to the saved agent file

    Returns:
        Dictionary with agent information
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Agent file not found: {filepath}")

    with open(filepath, 'rb') as f:
        agent_data = pickle.load(f)

    return {
        'agent_type': agent_data.get('agent_type', 'Unknown'),
        'player_id': agent_data.get('player_id'),
        'num_actions': agent_data.get('num_actions'),
        'q_table_size': len(agent_data.get('q_values', {})),
        'epsilon': agent_data.get('epsilon'),
        'file_size': os.path.getsize(filepath)
    }

def list_saved_agents(directory: str) -> list:
    """List all saved agents in a directory.

    Args:
        directory: Directory to search for saved agents

    Returns:
        List of dictionaries with agent information
    """
    if not os.path.exists(directory):
        return []

    agents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)
            try:
                info = get_agent_info(filepath)
                info['filename'] = filename
                info['filepath'] = filepath
                agents.append(info)
            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")

    return agents