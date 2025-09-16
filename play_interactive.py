#!/usr/bin/env python3
"""Interactive play script for OpenSpiel games with different agent types.

This script allows playing games with combinations of Human, Random, MCTS, and
trained Q-learner agents. It provides visual output and supports both
alternating and simultaneous move games.

Usage examples:
    # Human vs Random in cooperative box pushing
    python play_interactive.py --game coop_box_pushing --player0 human --player1 random

    # Q-learner vs MCTS in tic-tac-toe
    python play_interactive.py --game tic_tac_toe --player0 qlearner:agents/ttt_agent.pkl --player1 mcts

    # Human vs trained Q-learner
    python play_interactive.py --game coop_box_pushing --player0 human --player1 qlearner:agents/coop_agent.pkl
"""

import argparse
import os
import sys
import numpy as np
import pyspiel
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
from open_spiel.python.algorithms import tabular_qlearner

# Import our custom modules
from qlearner_bot import QLearnerBot
from save_load_agents import load_qlearner

class PlayerAwareHumanBot(pyspiel.Bot):
    """A human bot that knows its player ID for simultaneous move games."""

    def __init__(self, player_id):
        pyspiel.Bot.__init__(self)
        self._player_id = player_id
        self._human_bot = human.HumanBot()

    def step(self, state):
        """Choose an action for the given state."""
        # Get legal actions for this specific player
        if state.is_simultaneous_node():
            legal_actions = state.legal_actions(self._player_id)
        else:
            legal_actions = state.legal_actions()

        if not legal_actions:
            return pyspiel.INVALID_ACTION

        # Create action map for user input
        action_map = {}
        for action in legal_actions:
            try:
                action_str = state.action_to_string(self._player_id, action)
            except:
                action_str = str(action)
            action_map[action_str] = action
            action_map[str(action)] = action  # Also allow numeric input

        while True:
            print(f"\nPlayer {self._player_id} legal actions:")
            for action in legal_actions:
                try:
                    action_str = state.action_to_string(self._player_id, action)
                    print(f"  {action}: {action_str}")
                except:
                    print(f"  {action}")

            user_input = input(f"Player {self._player_id}, choose action: ").strip()

            if not user_input:
                continue

            # Try to parse as action string first
            if user_input in action_map:
                return action_map[user_input]

            # Try to parse as integer
            try:
                action = int(user_input)
                if action in legal_actions:
                    return action
                else:
                    print(f"Action {action} is not legal. Legal actions: {legal_actions}")
            except ValueError:
                print(f"Invalid input: {user_input}")

    def restart_at(self, state):
        pass

class GameVisualizer:
    """Handles visual display of game states."""

    @staticmethod
    def display_game_state(state, move_count=None, last_actions=None):
        """Display the current game state with nice formatting.

        Args:
            state: The current game state
            move_count: Optional move counter
            last_actions: Optional list of last actions taken
        """
        print("\n" + "="*60)
        if move_count is not None:
            print(f"Move {move_count}")
        print("="*60)

        # Display the state
        print(state)

        # Show last actions if available
        if last_actions:
            print(f"\nLast actions: {last_actions}")

        # Show current player(s)
        if hasattr(state, 'current_player'):
            current_player = state.current_player()
            if current_player >= 0:
                print(f"Current player: {current_player}")
            elif current_player == pyspiel.PlayerId.SIMULTANEOUS:
                print("Simultaneous move (all players act)")

        print("-"*60)

    @staticmethod
    def display_game_result(state):
        """Display the final game result.

        Args:
            state: The terminal game state
        """
        print("\n" + "🏁"*20)
        print("GAME OVER")
        print("🏁"*20)

        print("\nFinal state:")
        print(state)

        returns = state.returns()
        print(f"\nResults:")
        for i, utility in enumerate(returns):
            print(f"Player {i}: {utility:.2f}")

        total_utility = sum(returns)
        print(f"Total utility: {total_utility:.2f}")

        # Determine winner or cooperation success
        if len(returns) == 2:
            if abs(returns[0] - returns[1]) < 0.01:  # Tie
                if total_utility > 0:
                    print("🎉 Successful cooperation!")
                else:
                    print("🤝 Tie game")
            elif returns[0] > returns[1]:
                print("🏆 Player 0 wins!")
            else:
                print("🏆 Player 1 wins!")

def create_bot(bot_spec, player_id, game):
    """Create a bot from a specification string.

    Args:
        bot_spec: String specifying bot type and parameters
        player_id: The player ID for this bot
        game: The game instance

    Returns:
        A Bot instance
    """
    spec_parts = bot_spec.split(':', 1)
    bot_type = spec_parts[0].lower()

    if bot_type == 'human':
        return PlayerAwareHumanBot(player_id)

    elif bot_type == 'random':
        return uniform_random.UniformRandomBot(player_id, np.random)

    elif bot_type == 'mcts':
        # Use Information Set MCTS for imperfect information games
        if game.get_type().information == pyspiel.GameType.Information.IMPERFECT_INFORMATION:
            evaluator = pyspiel.RandomRolloutEvaluator(1, 42)
            return pyspiel.ISMCTSBot(
                42,  # seed
                evaluator,
                uct_c=2.0,
                max_simulations=1000,
                max_memory_mb=-1,
                final_policy_type=pyspiel.ISMCTSFinalPolicyType.NORMALIZED_VISIT_COUNT,
                solve=False,
                verbose=False
            )
        else:
            # Use regular MCTS for perfect information games
            evaluator = pyspiel.RandomRolloutEvaluator(1, 42)
            return pyspiel.MCTSBot(
                game,
                evaluator,
                uct_c=2.0,
                max_simulations=1000,
                max_memory_mb=-1,
                solve=False,
                seed=42,
                verbose=False
            )

    elif bot_type == 'qlearner':
        if len(spec_parts) != 2:
            raise ValueError("Q-learner bot requires path: qlearner:path/to/agent.pkl")

        agent_path = spec_parts[1]
        if not os.path.exists(agent_path):
            raise FileNotFoundError(f"Q-learner agent file not found: {agent_path}")

        # Load the trained Q-learner
        num_actions = game.num_distinct_actions()
        qlearner_agent = load_qlearner(agent_path, num_actions)

        # Verify player ID matches
        if qlearner_agent._player_id != player_id:
            print(f"Warning: Agent was trained for player {qlearner_agent._player_id}, "
                  f"but using for player {player_id}")

        return QLearnerBot(qlearner_agent, use_greedy=True)

    else:
        raise ValueError(f"Unknown bot type: {bot_type}. "
                        f"Supported types: human, random, mcts, qlearner:path")

def play_game(game, bots, verbose=True):
    """Play a complete game with the given bots.

    Args:
        game: The game instance
        bots: List of Bot instances
        verbose: Whether to show detailed output

    Returns:
        The final game state
    """
    state = game.new_initial_state()
    visualizer = GameVisualizer()
    move_count = 0

    if verbose:
        print(f"\nStarting game: {game.get_type().short_name}")
        print(f"Players: {game.num_players()}")
        for i, bot in enumerate(bots):
            bot_name = type(bot).__name__
            print(f"Player {i}: {bot_name}")

        visualizer.display_game_state(state, move_count)

    while not state.is_terminal():
        move_count += 1
        last_actions = []

        if state.is_chance_node():
            # Handle chance nodes
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            if verbose:
                print(f"Chance chose: {action} ({state.action_to_string(action)})")
            state.apply_action(action)
            last_actions = [f"Chance: {action}"]

        elif state.is_simultaneous_node():
            # Simultaneous moves
            actions = []
            for player_id in range(game.num_players()):
                if verbose:
                    print(f"\nPlayer {player_id}'s turn:")
                action = bots[player_id].step(state)
                actions.append(action)
                action_str = state.action_to_string(player_id, action)
                last_actions.append(f"P{player_id}: {action_str}")
                if verbose:
                    print(f"Player {player_id} chose: {action} ({action_str})")

            state.apply_actions(actions)

        else:
            # Sequential moves
            current_player = state.current_player()
            if verbose:
                print(f"\nPlayer {current_player}'s turn:")
            action = bots[current_player].step(state)
            action_str = state.action_to_string(current_player, action)
            last_actions.append(f"P{current_player}: {action_str}")
            if verbose:
                print(f"Player {current_player} chose: {action} ({action_str})")
            state.apply_action(action)

        if verbose:
            visualizer.display_game_state(state, move_count, last_actions)

    if verbose:
        visualizer.display_game_result(state)

    return state

def main():
    """Main function to run interactive play."""
    parser = argparse.ArgumentParser(
        description="Interactive play with different agent types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Agent types:
  human                    - Human player (interactive input)
  random                   - Random action selection
  mcts                     - Monte Carlo Tree Search
  qlearner:path/to/file    - Trained Q-learning agent

Examples:
  python play_interactive.py --game tic_tac_toe --player0 human --player1 mcts
  python play_interactive.py --game coop_box_pushing --player0 qlearner:agents/coop.pkl --player1 human
        """
    )

    parser.add_argument('--game', type=str, required=True,
                       help='Game to play (e.g., tic_tac_toe, coop_box_pushing)')
    parser.add_argument('--player0', type=str, required=True,
                       help='Bot type for player 0')
    parser.add_argument('--player1', type=str, required=True,
                       help='Bot type for player 1')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--num_games', type=int, default=1,
                       help='Number of games to play')

    args = parser.parse_args()

    try:
        # Load the game
        game = pyspiel.load_game(args.game)
        print(f"Loaded game: {game.get_type().short_name}")

        if game.num_players() != 2:
            print(f"Warning: This script is designed for 2-player games, "
                  f"but {args.game} has {game.num_players()} players")

        # Create bots
        bots = [
            create_bot(args.player0, 0, game),
            create_bot(args.player1, 1, game)
        ]

        # Play games
        results = []
        for game_num in range(args.num_games):
            if args.num_games > 1:
                print(f"\n{'='*20} GAME {game_num + 1} {'='*20}")

            final_state = play_game(game, bots, verbose=not args.quiet)
            results.append(final_state.returns())

            # Auto-continue to next game when running multiple games
            if args.num_games > 1 and game_num < args.num_games - 1:
                print(f"\nContinuing to game {game_num + 2}...")
                import time
                time.sleep(1)  # Brief pause for readability

        # Show summary for multiple games
        if args.num_games > 1:
            print(f"\n{'='*20} SUMMARY {'='*20}")
            player0_wins = sum(1 for r in results if r[0] > r[1])
            player1_wins = sum(1 for r in results if r[1] > r[0])
            ties = sum(1 for r in results if abs(r[0] - r[1]) < 0.01)

            print(f"Games played: {args.num_games}")
            print(f"Player 0 wins: {player0_wins}")
            print(f"Player 1 wins: {player1_wins}")
            print(f"Ties: {ties}")

            avg_returns = np.mean(results, axis=0)
            print(f"Average returns: P0={avg_returns[0]:.2f}, P1={avg_returns[1]:.2f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()