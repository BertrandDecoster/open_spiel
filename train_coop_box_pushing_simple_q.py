#!/usr/bin/env python3
"""Training script for simple Q-Learning on Cooperative Box Pushing.

This script trains two agents using independent Q-Learning to solve the
cooperative box pushing task with full observability.
"""

import numpy as np
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms.tabular_qlearner import QLearner

def create_environment():
    """Create the cooperative box pushing environment with full observability."""
    game = pyspiel.load_game('coop_box_pushing',
                            {'fully_observable': True, 'horizon': 100})
    env = rl_environment.Environment(game)
    return env

def create_agents(env):
    """Create two independent Q-Learning agents."""
    num_actions = env.action_spec()['num_actions']

    # Create epsilon schedule that decays over time
    epsilon_schedule = rl_tools.LinearSchedule(
        init_val=0.5,
        final_val=0.01,
        num_steps=5000  # Decay over first 5000 steps
    )

    # Create agents for both players
    agents = []
    for player_id in range(env.num_players):
        agent = QLearner(
            player_id=player_id,
            num_actions=num_actions,
            step_size=0.1,  # Learning rate
            epsilon_schedule=epsilon_schedule,
            discount_factor=0.99
        )
        agents.append(agent)

    return agents

def train_agents(env, agents, num_episodes=10000, eval_every=1000):
    """Train the agents using independent Q-Learning."""
    print(f"Starting training for {num_episodes} episodes...")
    print(f"Game has {env.num_players} players")

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for episode in range(num_episodes):
        time_step = env.reset()
        episode_reward = 0
        episode_length = 0

        while not time_step.last():
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(agents):
                step_output = agent.step(time_step, is_evaluation=False)
                actions.append(step_output.action)

            # Take step in environment
            time_step = env.step(actions)
            episode_reward += sum(time_step.rewards)
            episode_length += 1

        # Episode is over, step all agents with final info state for learning
        for agent in agents:
            agent.step(time_step, is_evaluation=False)

        # Episode finished
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Check if successful (positive reward indicates success)
        if episode_reward > 0:
            success_count += 1

        # Print progress
        if (episode + 1) % eval_every == 0:
            avg_reward = np.mean(episode_rewards[-eval_every:])
            avg_length = np.mean(episode_lengths[-eval_every:])
            success_rate = success_count / (episode + 1)

            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Average reward (last {eval_every}): {avg_reward:.3f}")
            print(f"  Average length (last {eval_every}): {avg_length:.1f}")
            print(f"  Success rate: {success_rate:.3f}")
            print(f"  Current epsilon: {agents[0]._epsilon:.3f}")
            print()

    return episode_rewards, episode_lengths, success_count

def evaluate_agents(env, agents, num_eval_episodes=100):
    """Evaluate the trained agents."""
    print(f"\nEvaluating agents for {num_eval_episodes} episodes...")

    eval_rewards = []
    eval_lengths = []
    eval_successes = 0

    for episode in range(num_eval_episodes):
        time_step = env.reset()
        episode_reward = 0
        episode_length = 0

        while not time_step.last():
            # Get greedy actions (no exploration)
            actions = []
            for i, agent in enumerate(agents):
                step_output = agent.step(time_step, is_evaluation=True)
                actions.append(step_output.action)

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

if __name__ == "__main__":
    # Create environment and agents
    env = create_environment()
    agents = create_agents(env)

    print("Cooperative Box Pushing - Independent Q-Learning Training")
    print("=" * 60)

    # Train the agents (reduced episodes for testing)
    episode_rewards, episode_lengths, success_count = train_agents(
        env, agents, num_episodes=1000, eval_every=200
    )

    # Final evaluation
    avg_reward, avg_length, success_rate = evaluate_agents(env, agents)

    print("\nTraining completed!")
    print(f"Final success rate: {success_rate:.3f}")
    print(f"Total training successes: {success_count}")