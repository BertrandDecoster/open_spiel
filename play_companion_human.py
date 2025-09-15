#!/usr/bin/env python3
"""Enhanced script to play companion games with human players and custom parameters."""

import sys
import os
from absl import app
from absl import flags
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build/python'))

import pyspiel
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random

# Game selection
_GAME_STRING = flags.DEFINE_enum(
    "game", "companion_simple", ["companion_simple", "companion_synchro"],
    "Which companion game to play"
)

# Game parameters
_ROWS = flags.DEFINE_integer("rows", 6, "Number of rows in the grid")
_COLS = flags.DEFINE_integer("cols", 6, "Number of columns in the grid")
_NUM_AGENTS = flags.DEFINE_integer("num_agents", 2, "Number of agents/players")
_HORIZON = flags.DEFINE_integer("horizon", 50, "Maximum game length")

# Player types
_PLAYER_TYPES = flags.DEFINE_list(
    "players", ["human", "human"],
    "Player types (human, random). Length must match num_agents."
)

def load_bot(bot_type: str, pid: int) -> pyspiel.Bot:
    """Load a bot of the specified type."""
    if bot_type == "human":
        return human.HumanBot()
    elif bot_type == "random":
        return uniform_random.UniformRandomBot(pid, np.random)
    else:
        raise ValueError(f"Unknown bot type: {bot_type}")

def print_game_info(game: pyspiel.Game):
    """Print information about the loaded game."""
    print(f"\n{'='*50}")
    print(f"🎮 Playing: {game.get_type().long_name}")
    print(f"📋 Parameters: {game.get_parameters()}")
    print(f"👥 Players: {game.num_players()}")
    print(f"🏁 Max game length: {game.max_game_length()}")
    print(f"{'='*50}\n")

def print_action_help():
    """Print help for companion game actions."""
    print("\n📝 Action Guide:")
    print("  • North/South/East/West: Move in that direction")
    print("  • Interact: Interact with objects at current position")
    print("  • Stay: Do nothing this turn")
    print("  • Type action name or press Enter to see legal actions")
    print()

def play_companion_game(state: pyspiel.State, bots: list[pyspiel.Bot]):
    """Play a companion game with the given bots."""

    print_action_help()
    step = 0

    while not state.is_terminal():
        print(f"\n--- Turn {step + 1} ---")
        print(f"Game State:\n{state}")

        if hasattr(state, 'returns'):
            returns = state.returns()
            print(f"💰 Current returns: {returns}")

        if state.is_simultaneous_node():
            # All players act simultaneously
            actions = []
            print(f"\n🔀 Simultaneous turn - all players choose actions:")

            for player in range(len(bots)):
                legal_actions = state.legal_actions(player)
                if legal_actions:
                    print(f"\n👤 Player {player} turn:")
                    action = bots[player].step(state)
                    if action in legal_actions:
                        action_str = state.action_to_string(player, action)
                        print(f"   Chose: {action_str}")
                        actions.append(action)
                    else:
                        print(f"   Invalid action {action}, using first legal action")
                        actions.append(legal_actions[0])
                else:
                    actions.append(0)  # Default action if no legal actions

            state.apply_actions(actions)

        else:
            # Sequential play (less common in companion games)
            player = state.current_player()
            legal_actions = state.legal_actions()

            if legal_actions:
                print(f"\n👤 Player {player} turn:")
                action = bots[player].step(state)
                action_str = state.action_to_string(action)
                print(f"   Chose: {action_str}")
                state.apply_action(action)

        step += 1

        # Add a small pause for readability
        if any(isinstance(bot, human.HumanBot) for bot in bots):
            input("\nPress Enter to continue...")

    # Game over
    print(f"\n{'='*50}")
    print("🏁 GAME OVER!")
    print(f"\nFinal state:\n{state}")

    if hasattr(state, 'returns'):
        returns = state.returns()
        print(f"\n🏆 Final scores: {returns}")

        # Determine winner(s)
        max_score = max(returns)
        winners = [i for i, score in enumerate(returns) if score == max_score]

        if len(winners) == 1:
            print(f"🥇 Winner: Player {winners[0]} with score {max_score}!")
        elif len(winners) == len(returns):
            print(f"🤝 It's a tie! All players scored {max_score}")
        else:
            winner_names = [f"Player {i}" for i in winners]
            print(f"🤝 Tie between {', '.join(winner_names)} with score {max_score}!")

    print(f"{'='*50}\n")

def main(_):
    """Main function to set up and play the game."""

    # Validate parameters
    if _NUM_AGENTS.value < 1:
        print("❌ Error: num_agents must be at least 1")
        return

    if len(_PLAYER_TYPES.value) != _NUM_AGENTS.value:
        print(f"❌ Error: Number of player types ({len(_PLAYER_TYPES.value)}) "
              f"must match num_agents ({_NUM_AGENTS.value})")
        print(f"   Use: --players={','.join(['human'] * _NUM_AGENTS.value)}")
        return

    # Set up game parameters
    params = {
        'rows': _ROWS.value,
        'cols': _COLS.value,
        'num_agents': _NUM_AGENTS.value,
        'horizon': _HORIZON.value
    }

    try:
        # Load the game
        game = pyspiel.load_game(_GAME_STRING.value, params)
        print_game_info(game)

        # Create bots
        bots = []
        for i, player_type in enumerate(_PLAYER_TYPES.value):
            bot = load_bot(player_type, i)
            bots.append(bot)
            print(f"👤 Player {i}: {player_type}")

        # Create initial state and play
        state = game.new_initial_state()
        play_companion_game(state, bots)

    except Exception as e:
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    print("🎮 Companion Games - Human Player")
    print("Usage examples:")
    print("  python play_companion_human.py --game=companion_simple --rows=4 --cols=4")
    print("  python play_companion_human.py --game=companion_synchro --num_agents=3 --players=human,human,random")
    print()
    app.run(main)