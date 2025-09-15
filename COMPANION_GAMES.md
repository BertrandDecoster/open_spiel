# Companion Games - Human Play Guide

This guide explains how to play the companion games (`companion_simple` and `companion_synchro`) as a human player with custom parameters.

## Quick Start

```bash
# Play companion_simple as human vs human on a 4x4 grid
python play_companion_human.py --game=companion_simple --rows=4 --cols=4

# Play companion_synchro with 3 players (2 human, 1 AI) on a 6x6 grid
python play_companion_human.py --game=companion_synchro --num_agents=3 --players=human,human,random --rows=6 --cols=6
```

## Game Parameters

Both companion games support the following parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rows` | 8 | Number of rows in the grid |
| `cols` | 8 | Number of columns in the grid |
| `num_agents` | 2 | Number of players/agents |
| `horizon` | 100 | Maximum game length |

## Player Types

- `human`: Interactive human player (you control via keyboard)
- `random`: AI that chooses random actions

## Game Actions

When playing as a human, you can choose from these actions:

- **North/South/East/West**: Move your agent in that direction
- **Interact**: Interact with objects at your current position
- **Stay**: Do nothing this turn (useful for coordination)

## Game Objectives

### Companion Simple
- **Goal**: Reach the goal cell (marked with `G`)
- **Reward**: All players get positive reward when any player reaches the goal
- **Strategy**: Coordinate to get at least one player to the goal quickly

### Companion Synchro
- **Goal**: Coordinate to activate synchro cells (marked with `S`)
- **Reward**: Positive reward when players work together at synchro locations
- **Strategy**: Coordinate movements to be at synchro cells simultaneously

## Examples

### Small 2-Player Game
```bash
python play_companion_human.py --game=companion_simple --rows=4 --cols=4 --horizon=20
```

### Large Multi-Player Game
```bash
python play_companion_human.py --game=companion_synchro --num_agents=4 --rows=8 --cols=8 --players=human,human,human,random
```

### Testing with AI Only
```bash
python play_companion_human.py --game=companion_simple --players=random,random --horizon=10
```

## Game Display

The games use ASCII art display:
- `^` = Agent (players)
- `G` = Goal (companion_simple)
- `S` = Synchro cell (companion_synchro)
- `.` = Empty space
- `#` = Wall/obstacle

Different colors are used for multiple agents when displayed.

## Tips for Human Play

1. **Read the state**: Always check the current grid layout before choosing actions
2. **Coordinate**: These are cooperative games - work together!
3. **Plan ahead**: Consider where other players are moving
4. **Use Stay**: Sometimes waiting is the best strategy for coordination
5. **Watch rewards**: Negative rewards each turn encourage quick solutions

## Using in Code

You can also use these games programmatically:

```python
import pyspiel
from open_spiel.python.bots import human

# Load with custom parameters
game = pyspiel.load_game('companion_simple', {
    'rows': 5,
    'cols': 5,
    'num_agents': 2,
    'horizon': 30
})

# Use human bots
bots = [human.HumanBot() for _ in range(game.num_players())]

# Play game...
```

## Troubleshooting

If you get import errors, make sure you're running from the OpenSpiel root directory and the build completed successfully:

```bash
cd /path/to/openspiel
./open_spiel/scripts/build_and_run_tests.sh
python play_companion_human.py
```