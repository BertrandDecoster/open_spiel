#!/usr/bin/env python3
"""Improved training script for Cooperative Box Pushing based on OpenSpiel tutorial.

This script follows the exact pattern from the OpenSpiel tutorial's Q-learning
implementation to fix the agent stepping and action passing issues.
"""

import numpy as np
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.algorithms import random_agent
from save_load_agents import save_multiple_agents

def create_environment():
    """Create the cooperative box pushing environment with full observability."""
    game = pyspiel.load_game('coop_box_pushing',
                            {'fully_observable': True, 'horizon': 100})
    env = rl_environment.Environment(game)
    return env

def create_agents(env):
    """Create Q-Learning agents following the tutorial pattern."""
    num_players = env.num_players
    num_actions = env.action_spec()["num_actions"]

    # Create agents exactly like the tutorial
    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    return agents

def train_agents(env, agents, num_episodes=1000, eval_every=200):
    """Train agents following the exact tutorial pattern."""
    print(f"Starting training for {num_episodes} episodes...")
    print(f"Game has {env.num_players} players")

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for cur_episode in range(num_episodes):
        if cur_episode % eval_every == 0:
            print(f"Episodes: {cur_episode}")

        time_step = env.reset()
        episode_reward = 0
        episode_length = 0

        while not time_step.last():
            # Check if this is an alternating move game or simultaneous
            current_player = time_step.observations.get("current_player", None)
            if current_player is not None and current_player >= 0:
                # Alternating moves (like tic_tac_toe) - current_player is 0, 1, 2, etc.
                player_id = current_player
                agent_output = agents[player_id].step(time_step)
                time_step = env.step([agent_output.action])
            else:
                # Simultaneous moves (like coop_box_pushing) - current_player is -2 or not present
                actions = []
                for agent in agents:
                    agent_output = agent.step(time_step)
                    actions.append(agent_output.action)
                time_step = env.step(actions)

            episode_reward += sum(time_step.rewards)
            episode_length += 1

        # CRITICAL: Episode is over, step all agents with final info state
        # This is the key pattern from the tutorial
        for agent in agents:
            agent.step(time_step)

        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode_reward > 0:
            success_count += 1

        # Print detailed progress
        if (cur_episode + 1) % eval_every == 0:
            avg_reward = np.mean(episode_rewards[-eval_every:])
            avg_length = np.mean(episode_lengths[-eval_every:])
            success_rate = success_count / (cur_episode + 1)

            print(f"Episode {cur_episode + 1}/{num_episodes}")
            print(f"  Average reward (last {eval_every}): {avg_reward:.3f}")
            print(f"  Average length (last {eval_every}): {avg_length:.1f}")
            print(f"  Success rate: {success_rate:.3f}")
            print(f"  Current epsilon: {agents[0]._epsilon:.3f}")
            print()

    print("Training Done!")
    return episode_rewards, episode_lengths, success_count

def evaluate_agents(env, agents, num_eval_episodes=100):
    """Evaluate trained agents against random agents."""
    print(f"\nEvaluating trained agents vs random for {num_eval_episodes} episodes...")

    # Create random agents for comparison
    num_actions = env.action_spec()["num_actions"]
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(env.num_players)
    ]

    eval_rewards = []
    eval_lengths = []
    eval_successes = 0

    for episode in range(num_eval_episodes):
        time_step = env.reset()
        episode_reward = 0
        episode_length = 0

        while not time_step.last():
            current_player = time_step.observations.get("current_player", None)
            if current_player is not None and current_player >= 0:
                # Alternating moves
                player_id = current_player
                agent_output = agents[player_id].step(time_step, is_evaluation=True)
                time_step = env.step([agent_output.action])
            else:
                # Simultaneous moves - use trained agents
                actions = []
                for agent in agents:
                    agent_output = agent.step(time_step, is_evaluation=True)
                    actions.append(agent_output.action)
                time_step = env.step(actions)

            episode_reward += sum(time_step.rewards)
            episode_length += 1

        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)

        if episode_reward > 0:
            eval_successes += 1

    avg_eval_reward = np.mean(eval_rewards)
    avg_eval_length = np.mean(eval_lengths)
    eval_success_rate = eval_successes / num_eval_episodes

    print("Evaluation Results:")
    print(f"  Average reward: {avg_eval_reward:.3f}")
    print(f"  Average episode length: {avg_eval_length:.1f}")
    print(f"  Success rate: {eval_success_rate:.3f}")

    return avg_eval_reward, avg_eval_length, eval_success_rate

def demo_play(env, agents):
    """Demo a single episode showing the trained agents."""
    print("\n=== Demo Play ===")
    time_step = env.reset()
    step_count = 0

    print("Initial state:")
    print(env.get_state)

    while not time_step.last() and step_count < 20:  # Limit for demo
        print(f"\nStep {step_count + 1}")

        current_player = time_step.observations.get("current_player", None)
        if current_player is not None and current_player >= 0:
            # Alternating moves
            player_id = current_player
            agent_output = agents[player_id].step(time_step, is_evaluation=True)
            print(f"Player {player_id} chooses action {agent_output.action}")
            time_step = env.step([agent_output.action])
        else:
            # Simultaneous moves
            actions = []
            for i, agent in enumerate(agents):
                agent_output = agent.step(time_step, is_evaluation=True)
                actions.append(agent_output.action)
                print(f"Player {i} chooses action {agent_output.action}")
            time_step = env.step(actions)

        step_count += 1
        if hasattr(env, 'get_state'):
            print(f"Rewards: {time_step.rewards}")

    print(f"\nFinal rewards: {time_step.rewards}")
    print("Demo complete!")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Train Q-Learning agents for cooperative box pushing')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--save_path', type=str, default='agents', help='Directory to save trained agents')
    parser.add_argument('--no_save', action='store_true', help='Skip saving agents')
    args = parser.parse_args()

    # Create environment and agents
    env = create_environment()
    agents = create_agents(env)

    print("Cooperative Box Pushing - Improved Q-Learning Training")
    print("Following OpenSpiel Tutorial Pattern")
    print("=" * 60)
    print(f"Training episodes: {args.episodes}")
    if not args.no_save:
        print(f"Will save agents to: {args.save_path}")

    # Train the agents
    episode_rewards, episode_lengths, success_count = train_agents(
        env, agents, num_episodes=args.episodes, eval_every=max(400, args.episodes//5)
    )

    # Final evaluation
    avg_reward, avg_length, success_rate = evaluate_agents(env, agents)

    # Save trained agents
    if not args.no_save:
        print(f"\nSaving trained agents...")
        try:
            saved_paths = save_multiple_agents(agents, args.save_path, "coop_box_pushing")
            print("Agents saved successfully:")
            for path in saved_paths:
                print(f"  {path}")

            print(f"\nTo play against these agents, use:")
            print(f"python play_interactive.py --game coop_box_pushing --player0 human --player1 qlearner:{saved_paths[1]}")
            print(f"python play_interactive.py --game coop_box_pushing --player0 qlearner:{saved_paths[0]} --player1 qlearner:{saved_paths[1]}")
        except Exception as e:
            print(f"Error saving agents: {e}")

    # Demo play
    demo_play(env, agents)

    print("\nTraining completed!")
    print(f"Final success rate: {success_rate:.3f}")
    print(f"Total training successes: {success_count}")