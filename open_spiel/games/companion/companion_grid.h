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

#ifndef OPEN_SPIEL_GAMES_COMPANION_GRID_H_
#define OPEN_SPIEL_GAMES_COMPANION_GRID_H_

#include <vector>
#include <utility>
#include <memory>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/companion/companion_types.h"

namespace open_spiel {
namespace companion {

// Grid class that manages the game state including the 2D grid of cells
// and all actors (agents, doors, ground items). Uses value semantics for
// efficient cloning in OpenSpiel's tree search algorithms.
class Grid {
 public:
  // Constructor for creating a grid with specified dimensions
  Grid(int rows, int cols);

  // Default constructor
  Grid() : Grid(8, 8) {}

  // Copy constructor and assignment (efficient for OpenSpiel cloning)
  Grid(const Grid& other) = default;
  Grid& operator=(const Grid& other) = default;

  // Destructor
  ~Grid() = default;

  // Grid dimensions
  int rows() const { return rows_; }
  int cols() const { return cols_; }

  // Fast cell access with bounds checking in debug mode
  CellType GetCell(int row, int col) const {
    SPIEL_DCHECK_GE(row, 0);
    SPIEL_DCHECK_LT(row, rows_);
    SPIEL_DCHECK_GE(col, 0);
    SPIEL_DCHECK_LT(col, cols_);
    return cells_[row * cols_ + col];
  }

  // Set cell type
  void SetCell(int row, int col, CellType type) {
    SPIEL_DCHECK_GE(row, 0);
    SPIEL_DCHECK_LT(row, rows_);
    SPIEL_DCHECK_GE(col, 0);
    SPIEL_DCHECK_LT(col, cols_);
    cells_[row * cols_ + col] = type;
  }

  // Bounds checking
  bool IsWithinBounds(int row, int col) const {
    return row >= 0 && row < rows_ && col >= 0 && col < cols_;
  }

  // Agent management
  void AddAgent(const AgentData& agent);
  void RemoveAgent(int agent_id);
  const AgentData* GetAgent(int agent_id) const;
  AgentData* GetAgent(int agent_id);
  const std::vector<AgentData>& GetAgents() const { return agents_; }

  // Get agent at specific position (returns nullptr if none)
  const AgentData* GetAgentAt(int row, int col) const;
  AgentData* GetAgentAt(int row, int col);

  // Get all agent IDs at a position
  std::vector<int> GetAgentIdsAt(int row, int col) const;

  // Door management
  void AddDoor(const DoorData& door);
  void RemoveDoor(int row, int col);
  const DoorData* GetDoorAt(int row, int col) const;
  DoorData* GetDoorAt(int row, int col);
  const std::vector<DoorData>& GetDoors() const { return doors_; }

  // Ground item management
  void AddGroundItem(const GroundItem& item);
  void RemoveGroundItem(int row, int col, int item_id);
  std::vector<GroundItem> GetGroundItemsAt(int row, int col) const;
  const std::vector<GroundItem>& GetGroundItems() const { return ground_items_; }

  // Movement and collision detection
  bool CanMoveTo(int row, int col) const;
  bool IsOverlappable(int row, int col) const;

  // Predict where agents would move given actions
  std::vector<std::pair<int, std::pair<int, int>>> PredictMoves(
      const std::vector<ActionType>& actions) const;

  // Resolve movement conflicts using the specified collision rules
  void ResolveCollisions(
      std::vector<std::pair<int, std::pair<int, int>>>& moves);

  // Apply valid moves to agents
  void ApplyMoves(const std::vector<std::pair<int, std::pair<int, int>>>& moves);

  // Handle interactions (doors, picking up items)
  void ProcessInteractions(const std::vector<ActionType>& actions);

  // Utility functions
  std::pair<int, int> FindEmptyCell() const;
  std::vector<std::pair<int, int>> FindEmptyCells() const;

  // Check if any agent is dead (in lava)
  std::vector<int> GetDeadAgents() const;

  // Remove dead agents from the grid
  void RemoveDeadAgents();

  // String representation for debugging
  std::string ToString() const;

 private:
  // Grid dimensions
  int rows_, cols_;

  // Flat array for cache-efficient cell access
  std::vector<CellType> cells_;

  // AOS for agents (efficient for small numbers of agents)
  std::vector<AgentData> agents_;

  // Other actors
  std::vector<DoorData> doors_;
  std::vector<GroundItem> ground_items_;

  // Helper functions
  int GetFlatIndex(int row, int col) const { return row * cols_ + col; }

  // Find agent index by ID
  int FindAgentIndex(int agent_id) const;

  // Check if a position contains non-overlappable objects
  bool HasNonOverlappableActor(int row, int col) const;
};

}  // namespace companion
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_COMPANION_GRID_H_