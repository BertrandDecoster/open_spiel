# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenSpiel is a framework for reinforcement learning in games, supporting n-player zero-sum, cooperative, and general-sum games. The core API and games are implemented in C++ and exposed to Python via pybind11. This is a fork of the official OpenSpiel repository from DeepMind.

## Build System and Development Commands

### Initial Setup
```bash
# One-time setup - install system dependencies and download external dependencies
./install.sh

# Set up Python virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Core Development Commands
```bash
# Build and test everything (main development command)
./open_spiel/scripts/build_and_run_tests.sh

# Build only without running tests
./open_spiel/scripts/build_and_run_tests.sh --build_only=true

# Build specific component (e.g., only Python bindings)
./open_spiel/scripts/build_and_run_tests.sh --test_only=python_test

# Build using pip (alternative approach)
./open_spiel/scripts/build_and_run_tests.sh --build_with_pip=true
```

### Manual Build Process
```bash
mkdir build && cd build
CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=${CXX} ../open_spiel
make -j$(nproc)
ctest -j$(nproc)

# Set PYTHONPATH for development
export PYTHONPATH=$PYTHONPATH:$(pwd)/..
export PYTHONPATH=$PYTHONPATH:$(pwd)/../open_spiel
export PYTHONPATH=$PYTHONPATH:$(pwd)/python
```

### Testing Specific Games
```bash
# Test a specific C++ component
./open_spiel/scripts/build_and_run_tests.sh --test_only=tic_tac_toe_test

# Test Python components
./open_spiel/scripts/build_and_run_tests.sh --test_only=python_test

# Run integration tests for all games
python3 open_spiel/integration_tests/playthrough_test.py

# Generate playthrough for new game
./open_spiel/scripts/generate_new_playthrough.sh game_name

# Regenerate all playthroughs after changes
./open_spiel/scripts/regenerate_playthroughs.sh
```

## Architecture and Code Organization

### Core Directory Structure
- `open_spiel/` - Core C++ API and game abstract classes
- `open_spiel/games/` - C++ game implementations
- `open_spiel/algorithms/` - C++ algorithms (CFR, MCTS, etc.)
- `open_spiel/python/` - Python API and implementations
- `open_spiel/python/algorithms/` - Python algorithms
- `open_spiel/python/examples/` - Python examples and demos
- `open_spiel/tests/` - C++ common test utilities
- `open_spiel/scripts/` - Build and development scripts
- `open_spiel/integration_tests/` - Generic tests for all games

### Key Architecture Components

**Core API (spiel.h)**:
- `Game` class: Represents game rules and configuration
- `State` class: Represents a game state with methods for legal actions, transitions
- `GameType` struct: Static information about game properties (sequential/simultaneous, perfect/imperfect information)
- `Observer` class: For getting observations of game states

**Game Implementation Pattern**:
Each game typically implements:
- `GameName` class inheriting from `Game`
- `GameNameState` class inheriting from `State`
- Registration macro `REGISTER_SPIEL_GAME`

**Python-C++ Bindings**:
- Core bindings in `open_spiel/python/pybind11/`
- Game-specific bindings in `games_[game_name].cc` files
- PySpiel module exposes C++ functionality to Python

### Game Types and Properties
Games are categorized by:
- **Dynamics**: Sequential, Simultaneous, or Mean Field
- **Information**: Perfect, Imperfect, or One-shot
- **Chance**: Deterministic, Explicit Stochastic, or Sampled Stochastic
- **Utility**: Zero-sum, Constant-sum, General-sum, or Identical

## Adding New Games

1. Choose a similar game to copy from `open_spiel/games/`
2. Copy header, source, and test files (e.g., `tic_tac_toe.h`, `tic_tac_toe.cc`, `tic_tac_toe_test.cc`)
3. Update `open_spiel/games/CMakeLists.txt` to include new files
4. Rename classes, namespaces, and identifiers
5. Update `open_spiel/python/tests/pyspiel_test.py` to include new game
6. Implement game logic in the new classes
7. Add playthrough file with `./open_spiel/scripts/generate_new_playthrough.sh game_name`
8. Run linting (cpplint for C++, pylint for Python)
9. Build and test to ensure integration

## Conditional Dependencies

OpenSpiel supports optional external dependencies controlled by environment variables:
- Set in `open_spiel/scripts/global_variables.sh`
- Examples: `OPEN_SPIEL_BUILD_WITH_HANABI=ON`, `OPEN_SPIEL_BUILD_WITH_JULIA=ON`
- Affects both installation (`install.sh`) and CMake build configuration
- Rerun `install.sh` after enabling new dependencies

## Common Development Patterns

**Python Virtual Environment Setup**:
The build system automatically manages virtual environments. Use `--virtualenv=false` to disable.

**CMake Configuration**:
- Prefers clang++ compiler
- Uses Python3_EXECUTABLE to specify Python interpreter
- BUILD_TYPE=Testing for development builds

**Testing Strategy**:
- C++ tests use Google Test framework
- Python tests in various `*_test.py` files
- Integration tests verify game mechanics via playthroughs
- Console play testing available via `ConsolePlayTest` utility

**Code Style**:
- Follows Google style guides
- C++: Use cpplint for linting
- Python: Use pylint with Google style guide configuration

## Language APIs

Multiple language bindings available:
- **Python**: Primary scripting interface (actively maintained)
- **Julia**: Available but requires `OPEN_SPIEL_BUILD_WITH_JULIA=ON`
- **Go**: Available but unmaintained
- **Rust**: Available but unmaintained

Python API closely mirrors C++ API, with many objects available in both languages for performance flexibility.

## Common Issues and Solutions

### Game Registration Issues in Python

**Problem**: C++ game builds successfully and tests pass, but game doesn't appear in `pyspiel.registered_games()`.

**Root Cause**: The game's `REGISTER_SPIEL_GAME` macro creates registration objects that may be eliminated by the linker's dead code optimization if not explicitly referenced.

**Solution**: Force linking by adding explicit references in Python bindings:

1. In the game's Python binding file (e.g., `games_companion_simple.cc`), add after includes:
```cpp
// Force linking of game registration
namespace open_spiel {
namespace companion {
extern std::shared_ptr<const Game> SimpleFactory(const GameParameters& params);
}
}
```

2. In the `init_pyspiel_games_*` function, add at the beginning:
```cpp
void open_spiel::init_pyspiel_games_companion_simple(py::module& m) {
  // Force reference to factory to ensure registration code is linked
  (void)companion::SimpleFactory;

  // ... rest of bindings
}
```

**Verification**: After rebuilding, check with:
```python
import pyspiel
print('game_name' in [g.short_name for g in pyspiel.registered_games()])
```

This pattern was used to fix the companion games (`companion_simple` and `companion_synchro`) registration issue.
- Current games use simple utility function, be we use an explicit way