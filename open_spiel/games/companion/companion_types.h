// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_COMPANION_TYPES_H_
#define OPEN_SPIEL_GAMES_COMPANION_TYPES_H_

#include <vector>

namespace open_spiel {
namespace companion {

// Types of cells in the grid
enum class CellType {
  kEmpty = 0,    // Agents can walk freely
  kWall = 1,     // No agents can move in
  kLava = 2,     // Agents die if pushed in, but can be pushed in
  kGoal = 3,     // Triggers game end (SimpleEnv)
  kSynchro = 4   // Synchronization points (SynchroEnv)
};

// Colors for visual distinction and key-door matching
enum class Color {
  kRed = 0,
  kBlue = 1,
  kGreen = 2,
  kYellow = 3,
  kPurple = 4,
  kOrange = 5,
  kCyan = 6,
  kPink = 7
};

// Cardinal directions for agent facing and movement
enum class Direction {
  kNorth = 0,
  kEast = 1,
  kSouth = 2,
  kWest = 3
};

// Actions available to agents
enum class ActionType {
  kNorth = 0,
  kEast = 1,
  kSouth = 2,
  kWest = 3,
  kInteract = 4,
  kStay = 5
};

// Types of agents
enum class AgentType {
  kRL = 0,   // Reinforcement Learning agent (or human-controlled)
  kFSM = 1   // Finite State Machine agent
};

// Pickable items that can be collected by agents
struct PickableItem {
  enum Type {
    kKey = 0,
    kTreasure = 1
  };

  Type type;
  Color color;
  int id;

  PickableItem() : type(kKey), color(Color::kRed), id(0) {}
  PickableItem(Type t, Color c, int i) : type(t), color(c), id(i) {}

  bool operator==(const PickableItem& other) const {
    return type == other.type && color == other.color && id == other.id;
  }
};

// Agent data structure using AOS for performance
struct AgentData {
  int id;                                // Unique agent identifier
  int row, col;                          // Position on the grid
  Direction direction;                   // Facing direction
  Color color;                           // Visual distinction
  AgentType type;                        // RL or FSM agent
  std::vector<PickableItem> inventory;   // Collected items

  // FSM-specific data (only used when type == kFSM)
  int fsm_state;                         // Current FSM state
  int patrol_route_id;                   // Which patrol route to follow

  AgentData()
      : id(0), row(0), col(0), direction(Direction::kNorth),
        color(Color::kRed), type(AgentType::kRL),
        fsm_state(0), patrol_route_id(0) {
    inventory.reserve(4);  // Typical max inventory size
  }

  AgentData(int agent_id, int r, int c, Direction dir, Color col, AgentType t)
      : id(agent_id), row(r), col(c), direction(dir), color(col), type(t),
        fsm_state(0), patrol_route_id(0) {
    inventory.reserve(4);
  }

  // Check if agent has a specific key
  bool HasKey(Color key_color) const {
    for (const auto& item : inventory) {
      if (item.type == PickableItem::kKey && item.color == key_color) {
        return true;
      }
    }
    return false;
  }

  // Remove a key from inventory (returns true if found and removed)
  bool RemoveKey(Color key_color) {
    for (auto it = inventory.begin(); it != inventory.end(); ++it) {
      if (it->type == PickableItem::kKey && it->color == key_color) {
        inventory.erase(it);
        return true;
      }
    }
    return false;
  }
};

// Door data structure
struct DoorData {
  int row, col;                    // Position on the grid
  Color color;                     // Visual color of the door
  Color required_key_color;        // Which key color opens this door
  bool is_open;                    // Current state

  DoorData()
      : row(0), col(0), color(Color::kRed),
        required_key_color(Color::kRed), is_open(false) {}

  DoorData(int r, int c, Color door_color, Color key_color, bool open = false)
      : row(r), col(c), color(door_color),
        required_key_color(key_color), is_open(open) {}
};

// Ground item data structure (items lying on the ground to be picked up)
struct GroundItem {
  int row, col;                    // Position on the grid
  PickableItem item;               // The actual item

  GroundItem() : row(0), col(0) {}
  GroundItem(int r, int c, const PickableItem& i)
      : row(r), col(c), item(i) {}
};

// Movement deltas for each direction
constexpr int kRowDeltas[] = {-1, 0, 1, 0};  // North, East, South, West
constexpr int kColDeltas[] = {0, 1, 0, -1};  // North, East, South, West

// Game constants
constexpr int kNumActions = 6;  // 4 moves + interact + stay
constexpr int kMaxAgents = 25;  // 3 RL + up to 22 FSM agents
constexpr int kMaxInventorySize = 8;

// Utility functions
inline Direction RotateLeft(Direction dir) {
  return static_cast<Direction>((static_cast<int>(dir) + 3) % 4);
}

inline Direction RotateRight(Direction dir) {
  return static_cast<Direction>((static_cast<int>(dir) + 1) % 4);
}

inline std::pair<int, int> GetNextPosition(int row, int col, Direction dir) {
  return {row + kRowDeltas[static_cast<int>(dir)],
          col + kColDeltas[static_cast<int>(dir)]};
}

}  // namespace companion
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_COMPANION_TYPES_H_