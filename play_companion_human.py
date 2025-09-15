#!/usr/bin/env python3
"""Enhanced script to play companion games with human players and custom parameters."""

import sys
import os
from absl import app
from absl import flags
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build/python'))

# ANSI color codes for terminal output
class Colors:
    # Regular colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'

    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    # Grayscale
    GRAY = '\033[90m'

# Mapping from C++ Color enum values to terminal colors
AGENT_COLOR_MAP = {
    0: Colors.BRIGHT_RED,     # Color::kRed
    1: Colors.BRIGHT_BLUE,    # Color::kBlue
    2: Colors.BRIGHT_GREEN,   # Color::kGreen
    3: Colors.BRIGHT_YELLOW,  # Color::kYellow
    4: Colors.BRIGHT_MAGENTA, # Color::kPurple
    5: Colors.YELLOW,         # Color::kOrange
    6: Colors.BRIGHT_CYAN,    # Color::kCyan
    7: Colors.MAGENTA,        # Color::kPink
}

import pyspiel
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random

# Import companion enums for comparison
try:
    from open_spiel.python.games.companion_base import CellType
except ImportError:
    # Fallback for different import paths
    try:
        import pyspiel
        CellType = pyspiel.companion_base.CellType
    except:
        CellType = None

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

# Display options
_USE_COLORS = flags.DEFINE_boolean("colors", True, "Use colors in the display")
_USE_ARROW_KEYS = flags.DEFINE_boolean("arrow_keys", True, "Use arrow keys for movement (requires compatible terminal)")

def get_arrow_key_input():
    """Get single keypress input including arrow keys. Returns action string or None."""
    try:
        import termios
        import tty
        import sys

        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)

            # Handle escape sequences (arrow keys)
            if ch == '\x1b':  # ESC
                ch += sys.stdin.read(2)
                if ch == '\x1b[A':  # Up arrow
                    return 'North'
                elif ch == '\x1b[B':  # Down arrow
                    return 'South'
                elif ch == '\x1b[C':  # Right arrow
                    return 'East'
                elif ch == '\x1b[D':  # Left arrow
                    return 'West'
            elif ch == ' ':  # Space
                return 'Interact'
            elif ch == '\r' or ch == '\n':  # Enter
                return 'Stay'
            elif ch == '\x03':  # Ctrl+C
                raise KeyboardInterrupt
            elif ch == 'q' or ch == 'Q':
                raise KeyboardInterrupt

        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        return None
    except (ImportError, OSError):
        # Fall back to regular input if termios not available
        return None


class SimultaneousHumanBot:
    """Human bot wrapper for simultaneous games."""

    def __init__(self, player_id, use_arrow_keys=False):
        self.player_id = player_id
        self.use_arrow_keys = use_arrow_keys

    def step(self, state):
        """Get action from human player for simultaneous games."""
        legal_actions = state.legal_actions(self.player_id)
        if not legal_actions:
            return 0  # Default action if no legal actions

        # Create mapping from full action names to action IDs
        action_map = {
            state.action_to_string(self.player_id, action): action
            for action in legal_actions
        }

        # Add short aliases for common actions
        short_aliases = {}
        for action in legal_actions:
            action_name = state.action_to_string(self.player_id, action)
            if action_name == "North":
                short_aliases.update({"n": action, "N": action, "1": action})
            elif action_name == "East":
                short_aliases.update({"e": action, "E": action, "2": action})
            elif action_name == "South":
                short_aliases.update({"s": action, "S": action, "3": action})
            elif action_name == "West":
                short_aliases.update({"w": action, "W": action, "4": action})
            elif action_name == "Interact":
                short_aliases.update({"i": action, "I": action, "5": action})
            elif action_name == "Stay":
                short_aliases.update({"x": action, "X": action, "6": action, ".": action})

        # Combine both mappings
        action_map.update(short_aliases)

        # Try arrow key input first if enabled
        if self.use_arrow_keys:
            print("Use arrow keys to move, Space to interact, Enter to stay, or type action:")
            action_str = get_arrow_key_input()
            if action_str and action_str in action_map:
                return action_map[action_str]

        while True:
            if self.use_arrow_keys:
                action_str = input("Or type action (empty to print legal actions): ")
            else:
                action_str = input("Choose an action (empty to print legal actions): ")

            if not action_str:
                print("Legal action(s):")
                for i, action in enumerate(legal_actions, 1):
                    action_name = state.action_to_string(self.player_id, action)
                    shortcuts = []
                    if action_name == "North":
                        shortcuts = ["N/n", "1", "↑"] if self.use_arrow_keys else ["N/n", "1"]
                    elif action_name == "East":
                        shortcuts = ["E/e", "2", "→"] if self.use_arrow_keys else ["E/e", "2"]
                    elif action_name == "South":
                        shortcuts = ["S/s", "3", "↓"] if self.use_arrow_keys else ["S/s", "3"]
                    elif action_name == "West":
                        shortcuts = ["W/w", "4", "←"] if self.use_arrow_keys else ["W/w", "4"]
                    elif action_name == "Interact":
                        shortcuts = ["I/i", "5", "Space"] if self.use_arrow_keys else ["I/i", "5"]
                    elif action_name == "Stay":
                        shortcuts = ["X/x", ".", "6", "Enter"] if self.use_arrow_keys else ["X/x", ".", "6"]

                    shortcut_str = f" ({', '.join(shortcuts)})" if shortcuts else ""
                    print(f"  {action_name}{shortcut_str}")
                continue

            if action_str in action_map:
                return action_map[action_str]

            print(f"Invalid action '{action_str}'. Try again.")


def colorize_char(char: str, use_colors: bool = True, agent_color_id: int = -1) -> str:
    """Add color to game characters based on their meaning."""
    if not use_colors:
        return char

    # Agent direction symbols or numbers (colored by actual agent color)
    if (char in '^>v<' or char.isdigit()) and agent_color_id >= 0:
        agent_color = AGENT_COLOR_MAP.get(agent_color_id, Colors.WHITE)
        return agent_color + char + Colors.RESET

    # Map other symbols
    color_map = {
        'G': Colors.BRIGHT_GREEN + 'G' + Colors.RESET,      # Goal
        '~': Colors.BRIGHT_RED + '~' + Colors.RESET,        # Lava
        '#': Colors.GRAY + '#' + Colors.RESET,              # Wall
        'S': Colors.BRIGHT_CYAN + 'S' + Colors.RESET,       # Synchro
        '*': Colors.BRIGHT_YELLOW + '*' + Colors.RESET,     # Items
        '+': Colors.YELLOW + '+' + Colors.RESET,            # Closed door
        '/': Colors.GREEN + '/' + Colors.RESET,             # Open door
        '.': Colors.DIM + '.' + Colors.RESET,               # Empty space
    }

    return color_map.get(char, char)


def colorize_game_state_with_agent_colors(state, use_colors: bool = True) -> str:
    """Colorize the game state using actual agent color information from the grid."""
    if not use_colors:
        return str(state)

    try:
        # Get the grid from the state
        grid = state.get_grid()
        rows = grid.rows()
        cols = grid.cols()

        # Build the colorized grid
        colorized_grid_lines = []
        for row in range(rows):
            line_chars = []
            for col in range(cols):
                # Check for multiple agents at this position
                agent_ids = grid.get_agent_ids_at(row, col)

                if agent_ids:
                    # Show the first agent, but indicate if there are multiple
                    display_char, agent_color_id = grid.get_agent_display_char(row, col)

                    if len(agent_ids) > 1:
                        # Multiple agents - show number instead of direction
                        char_to_show = str(len(agent_ids))
                        line_chars.append(colorize_char(char_to_show, use_colors, agent_color_id))
                    else:
                        # Single agent - show direction with color
                        line_chars.append(colorize_char(display_char, use_colors, agent_color_id))
                else:
                    # No agents - check for other game elements
                    cell_type = grid.get_cell(row, col)

                    # Check for doors
                    door = grid.get_door_at(row, col)
                    if door is not None:
                        door_char = '/' if door.is_open else '+'
                        line_chars.append(colorize_char(door_char, use_colors))
                    else:
                        # Check for ground items
                        items = grid.get_ground_items_at(row, col)
                        if items:
                            line_chars.append(colorize_char('*', use_colors))
                        else:
                            # Show cell type using enum comparison
                            if CellType and cell_type == CellType.EMPTY:
                                line_chars.append(colorize_char('.', use_colors))
                            elif CellType and cell_type == CellType.WALL:
                                line_chars.append(colorize_char('#', use_colors))
                            elif CellType and cell_type == CellType.LAVA:
                                line_chars.append(colorize_char('~', use_colors))
                            elif CellType and cell_type == CellType.GOAL:
                                line_chars.append(colorize_char('G', use_colors))
                            elif CellType and cell_type == CellType.SYNCHRO:
                                line_chars.append(colorize_char('S', use_colors))
                            else:
                                # Fallback for integer comparison or unknown types
                                if cell_type == 0:  # kEmpty
                                    line_chars.append(colorize_char('.', use_colors))
                                elif cell_type == 1:  # kWall
                                    line_chars.append(colorize_char('#', use_colors))
                                elif cell_type == 2:  # kLava
                                    line_chars.append(colorize_char('~', use_colors))
                                elif cell_type == 3:  # kGoal
                                    line_chars.append(colorize_char('G', use_colors))
                                elif cell_type == 4:  # kSynchro
                                    line_chars.append(colorize_char('S', use_colors))
                                else:
                                    line_chars.append('.')

            colorized_grid_lines.append(''.join(line_chars))

        # Build the full state string with headers
        state_lines = []

        # Add the header info with colors
        if hasattr(state, 'get_grid'):
            if 'Simple' in str(type(state)):
                state_lines.append(Colors.BOLD + Colors.CYAN + 'CompanionSimple State:' + Colors.RESET)
            else:
                state_lines.append(Colors.BOLD + Colors.CYAN + 'CompanionSynchro State:' + Colors.RESET)

        # Add timestep and other info
        # Note: We'll parse this from the original string since we don't have direct access
        original_lines = str(state).split('\n')
        for line in original_lines:
            if line.startswith('Timestep:'):
                state_lines.append(line)
            elif line.startswith('Terminal:'):
                state_lines.append(line)
            elif 'Returns:' in line:
                state_lines.append(Colors.BRIGHT_YELLOW + line + Colors.RESET)

        # Add empty line before grid
        state_lines.append('')

        # Add the colorized grid
        state_lines.extend(colorized_grid_lines)

        return '\n'.join(state_lines)

    except Exception as e:
        # Fallback to simple string colorization if grid access fails
        return colorize_game_state_fallback(str(state), use_colors)


def colorize_game_state_fallback(state_str: str, use_colors: bool = True) -> str:
    """Fallback colorization method using string processing."""
    if not use_colors:
        return state_str

    # Split into lines and process each character
    lines = state_str.split('\n')
    colorized_lines = []

    for line in lines:
        if any(c in line for c in '^>v<G~#S*+/.'):
            # This looks like a grid line - colorize each character
            colorized_line = ''.join(colorize_char(c, use_colors) for c in line)
            colorized_lines.append(colorized_line)
        else:
            # Keep non-grid lines as-is but add some color to headers
            if line.startswith('CompanionSimple State:') or line.startswith('CompanionSynchro State:'):
                colorized_lines.append(Colors.BOLD + Colors.CYAN + line + Colors.RESET)
            elif 'Returns:' in line:
                colorized_lines.append(Colors.BRIGHT_YELLOW + line + Colors.RESET)
            else:
                colorized_lines.append(line)

    return '\n'.join(colorized_lines)


def load_bot(bot_type: str, pid: int, use_arrow_keys: bool = False) -> pyspiel.Bot:
    """Load a bot of the specified type."""
    if bot_type == "human":
        return SimultaneousHumanBot(pid, use_arrow_keys)
    elif bot_type == "random":
        return uniform_random.UniformRandomBot(pid, np.random)
    else:
        raise ValueError(f"Unknown bot type: {bot_type}")

def print_game_info(game: pyspiel.Game):
    """Print information about the loaded game."""
    print(f"\n{Colors.CYAN}{'='*50}{Colors.RESET}")
    print(f"🎮 {Colors.BOLD}Playing: {game.get_type().long_name}{Colors.RESET}")
    print(f"📋 Parameters: {game.get_parameters()}")
    print(f"👥 Players: {game.num_players()}")
    print(f"🏁 Max game length: {game.max_game_length()}")
    print(f"{Colors.CYAN}{'='*50}{Colors.RESET}\n")

def print_action_help():
    """Print help for companion game actions."""
    print(f"\n{Colors.BOLD}📝 Action Guide:{Colors.RESET}")
    print(f"  • {Colors.BRIGHT_CYAN}Movement:{Colors.RESET} North/South/East/West (or N/S/E/W, 1/2/3/4)")
    print(f"  • {Colors.BRIGHT_YELLOW}Interact:{Colors.RESET} Interact with objects (I/i, 5)")
    print(f"  • {Colors.BRIGHT_GREEN}Stay:{Colors.RESET} Do nothing this turn (X/x, ., 6)")
    print(f"  • {Colors.DIM}Press Enter to see available actions{Colors.RESET}")
    print()

    # Print legend for game symbols
    print(f"{Colors.BOLD}🎯 Symbol Legend:{Colors.RESET}")
    print(f"  {Colors.BRIGHT_RED}^>v<{Colors.RESET} - Single player (facing direction)  {Colors.BRIGHT_RED}2,3,4{Colors.RESET} - Multiple players (count)")
    print(f"  {Colors.BRIGHT_GREEN}G{Colors.RESET} - Goal  {Colors.GRAY}#{Colors.RESET} - Wall  {Colors.BRIGHT_RED}~{Colors.RESET} - Lava")
    print(f"  {Colors.BRIGHT_CYAN}S{Colors.RESET} - Synchro  {Colors.BRIGHT_YELLOW}*{Colors.RESET} - Items  {Colors.DIM}.{Colors.RESET} - Empty")
    print()

def play_companion_game(state: pyspiel.State, bots: list[pyspiel.Bot], use_colors: bool = True):
    """Play a companion game with the given bots."""

    print_action_help()
    step = 0

    while not state.is_terminal():
        print(f"\n{Colors.BOLD}{Colors.BLUE}--- Turn {step + 1} ---{Colors.RESET}")
        game_state_str = colorize_game_state_with_agent_colors(state, use_colors)
        print(f"Game State:\n{game_state_str}")

        if hasattr(state, 'returns'):
            returns = state.returns()
            print(f"💰 {Colors.BRIGHT_YELLOW}Current returns: {returns}{Colors.RESET}")

        if state.is_simultaneous_node():
            # All players act simultaneously
            actions = []
            print(f"\n🔀 Simultaneous turn - all players choose actions:")

            for player in range(len(bots)):
                legal_actions = state.legal_actions(player)
                if legal_actions:
                    # Use the actual agent color from the game
                    player_color = AGENT_COLOR_MAP.get(player % len(AGENT_COLOR_MAP), Colors.WHITE)
                    print(f"\n👤 {player_color}Player {player}{Colors.RESET} turn:")
                    action = bots[player].step(state)
                    if action in legal_actions:
                        action_str = state.action_to_string(player, action)
                        print(f"   Chose: {Colors.BRIGHT_WHITE}{action_str}{Colors.RESET}")
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
    print(f"\n{Colors.CYAN}{'='*50}{Colors.RESET}")
    print(f"🏁 {Colors.BOLD}{Colors.BRIGHT_YELLOW}GAME OVER!{Colors.RESET}")
    final_state_str = colorize_game_state_with_agent_colors(state, use_colors)
    print(f"\nFinal state:\n{final_state_str}")

    if hasattr(state, 'returns'):
        returns = state.returns()
        print(f"\n🏆 {Colors.BRIGHT_YELLOW}Final scores: {returns}{Colors.RESET}")

        # Determine winner(s)
        max_score = max(returns)
        winners = [i for i, score in enumerate(returns) if score == max_score]

        if len(winners) == 1:
            # Use the actual agent color from the game
            winner_color = AGENT_COLOR_MAP.get(winners[0] % len(AGENT_COLOR_MAP), Colors.WHITE)
            print(f"🥇 {Colors.BOLD}Winner: {winner_color}Player {winners[0]}{Colors.RESET}{Colors.BOLD} with score {max_score}!{Colors.RESET}")
        elif len(winners) == len(returns):
            print(f"🤝 {Colors.BRIGHT_CYAN}It's a tie! All players scored {max_score}{Colors.RESET}")
        else:
            winner_names = [f"Player {i}" for i in winners]
            print(f"🤝 {Colors.BRIGHT_CYAN}Tie between {', '.join(winner_names)} with score {max_score}!{Colors.RESET}")

    print(f"{Colors.CYAN}{'='*50}{Colors.RESET}\n")

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
            bot = load_bot(player_type, i, _USE_ARROW_KEYS.value)
            bots.append(bot)
            if _USE_COLORS.value:
                # Use the actual agent color from the game
                player_color = AGENT_COLOR_MAP.get(i % len(AGENT_COLOR_MAP), Colors.WHITE)
                print(f"👤 {player_color}Player {i}{Colors.RESET}: {player_type}")
            else:
                print(f"👤 Player {i}: {player_type}")

        # Show input mode information
        if _USE_ARROW_KEYS.value:
            print(f"\n{Colors.BRIGHT_CYAN}🎮 Arrow key mode enabled! Use ←↑↓→ keys for movement.{Colors.RESET}")

        # Create initial state and play
        state = game.new_initial_state()
        play_companion_game(state, bots, _USE_COLORS.value)

    except Exception as e:
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    print("🎮 Companion Games - Enhanced Human Player Interface")
    print("Usage examples:")
    print("  # Basic 2-player game")
    print("  python play_companion_human.py --game=companion_simple --rows=4 --cols=4")
    print("  # 3-player mixed game")
    print("  python play_companion_human.py --num_agents=3 --players=human,human,random")
    print("  # 4-player synchro game")
    print("  python play_companion_human.py --game=companion_synchro --num_agents=4 --players=human,random,random,random")
    print("  # Disable arrow keys or colors")
    print("  python play_companion_human.py --noarrow_keys --nocolor")
    print()
    print("Player configuration:")
    print("  --num_agents=N        Number of players (2-8)")
    print("  --players=type1,type2 Player types: human, random")
    print("  Example: --num_agents=4 --players=human,human,random,random")
    print()
    print("Features:")
    print("  • Arrow key controls: ←↑↓→ movement, Space=interact, Enter=stay (default ON)")
    print("  • Short action names: N/S/E/W, I (interact), X (stay), 1-6 (numeric)")
    print("  • Agent-based colors: Each player has their unique assigned color")
    print("  • Direction tracking: Arrows (^>v<) update as agents face different directions")
    print()
    app.run(main)