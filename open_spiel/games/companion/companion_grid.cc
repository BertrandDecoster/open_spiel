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

#include "open_spiel/games/companion/companion_grid.h"

#include <algorithm>
#include <sstream>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace companion {

Grid::Grid(int rows, int cols) : rows_(rows), cols_(cols) {
  cells_.resize(rows * cols, CellType::kEmpty);
  agents_.reserve(kMaxAgents);
  doors_.reserve(16);
  ground_items_.reserve(32);
}

void Grid::AddAgent(const AgentData& agent) {
  // Check if agent ID already exists
  if (GetAgent(agent.id) != nullptr) {
    SpielFatalError(absl::StrCat("Agent with ID ", agent.id, " already exists"));
  }
  agents_.push_back(agent);
}

void Grid::RemoveAgent(int agent_id) {
  int index = FindAgentIndex(agent_id);
  if (index >= 0) {
    agents_.erase(agents_.begin() + index);
  }
}

const AgentData* Grid::GetAgent(int agent_id) const {
  int index = FindAgentIndex(agent_id);
  return index >= 0 ? &agents_[index] : nullptr;
}

AgentData* Grid::GetAgent(int agent_id) {
  int index = FindAgentIndex(agent_id);
  return index >= 0 ? &agents_[index] : nullptr;
}

const AgentData* Grid::GetAgentAt(int row, int col) const {
  for (const auto& agent : agents_) {
    if (agent.row == row && agent.col == col) {
      return &agent;
    }
  }
  return nullptr;
}

AgentData* Grid::GetAgentAt(int row, int col) {
  for (auto& agent : agents_) {
    if (agent.row == row && agent.col == col) {
      return &agent;
    }
  }
  return nullptr;
}

std::vector<int> Grid::GetAgentIdsAt(int row, int col) const {
  std::vector<int> agent_ids;
  for (const auto& agent : agents_) {
    if (agent.row == row && agent.col == col) {
      agent_ids.push_back(agent.id);
    }
  }
  return agent_ids;
}

void Grid::AddDoor(const DoorData& door) {
  doors_.push_back(door);
}

void Grid::RemoveDoor(int row, int col) {
  doors_.erase(
      std::remove_if(doors_.begin(), doors_.end(),
                     [row, col](const DoorData& door) {
                       return door.row == row && door.col == col;
                     }),
      doors_.end());
}

const DoorData* Grid::GetDoorAt(int row, int col) const {
  for (const auto& door : doors_) {
    if (door.row == row && door.col == col) {
      return &door;
    }
  }
  return nullptr;
}

DoorData* Grid::GetDoorAt(int row, int col) {
  for (auto& door : doors_) {
    if (door.row == row && door.col == col) {
      return &door;
    }
  }
  return nullptr;
}

void Grid::AddGroundItem(const GroundItem& item) {
  ground_items_.push_back(item);
}

void Grid::RemoveGroundItem(int row, int col, int item_id) {
  ground_items_.erase(
      std::remove_if(ground_items_.begin(), ground_items_.end(),
                     [row, col, item_id](const GroundItem& item) {
                       return item.row == row && item.col == col &&
                              item.item.id == item_id;
                     }),
      ground_items_.end());
}

std::vector<GroundItem> Grid::GetGroundItemsAt(int row, int col) const {
  std::vector<GroundItem> items;
  for (const auto& item : ground_items_) {
    if (item.row == row && item.col == col) {
      items.push_back(item);
    }
  }
  return items;
}

bool Grid::CanMoveTo(int row, int col) const {
  if (!IsWithinBounds(row, col)) {
    return false;
  }

  CellType cell = GetCell(row, col);
  if (cell == CellType::kWall) {
    return false;
  }

  // Check for closed doors
  const DoorData* door = GetDoorAt(row, col);
  if (door != nullptr && !door->is_open) {
    return false;
  }

  return true;
}

bool Grid::IsOverlappable(int row, int col) const {
  if (!IsWithinBounds(row, col)) {
    return false;
  }

  // Lava is overlappable (agents can be pushed in)
  CellType cell = GetCell(row, col);
  if (cell == CellType::kWall) {
    return false;
  }

  // Closed doors are not overlappable
  const DoorData* door = GetDoorAt(row, col);
  if (door != nullptr && !door->is_open) {
    return false;
  }

  // Two agents cannot overlap
  if (GetAgentAt(row, col) != nullptr) {
    return false;
  }

  return true;
}

std::vector<std::pair<int, std::pair<int, int>>> Grid::PredictMoves(
    const std::vector<ActionType>& actions) const {
  std::vector<std::pair<int, std::pair<int, int>>> moves;
  moves.reserve(agents_.size());

  for (size_t i = 0; i < agents_.size() && i < actions.size(); ++i) {
    const AgentData& agent = agents_[i];
    ActionType action = actions[i];

    int new_row = agent.row;
    int new_col = agent.col;

    // Calculate new position based on action
    if (action == ActionType::kNorth ||
        action == ActionType::kEast ||
        action == ActionType::kSouth ||
        action == ActionType::kWest) {
      Direction move_dir = static_cast<Direction>(static_cast<int>(action));
      std::tie(new_row, new_col) = GetNextPosition(agent.row, agent.col, move_dir);
    }

    moves.emplace_back(agent.id, std::make_pair(new_row, new_col));
  }

  return moves;
}

void Grid::ResolveCollisions(
    std::vector<std::pair<int, std::pair<int, int>>>& moves) {
  // First pass: Check if moves are valid (within bounds, not walls, etc.)
  for (auto& move : moves) {
    int agent_id = move.first;
    int new_row = move.second.first;
    int new_col = move.second.second;

    const AgentData* agent = GetAgent(agent_id);
    if (agent == nullptr) continue;

    // If move is invalid, agent stays in place
    if (!CanMoveTo(new_row, new_col)) {
      move.second = {agent->row, agent->col};
    }
  }

  // Second pass: Resolve agent-agent collisions
  for (size_t i = 0; i < moves.size(); ++i) {
    for (size_t j = i + 1; j < moves.size(); ++j) {
      int agent1_id = moves[i].first;
      int agent2_id = moves[j].first;
      auto& pos1 = moves[i].second;
      auto& pos2 = moves[j].second;

      // If two agents try to move to the same position
      if (pos1.first == pos2.first && pos1.second == pos2.second) {
        const AgentData* agent1 = GetAgent(agent1_id);
        const AgentData* agent2 = GetAgent(agent2_id);

        if (agent1 == nullptr || agent2 == nullptr) continue;

        bool agent1_moved = (pos1.first != agent1->row || pos1.second != agent1->col);
        bool agent2_moved = (pos2.first != agent2->row || pos2.second != agent2->col);

        if (agent1_moved && agent2_moved) {
          // Both agents moved to same position - both bounce back
          pos1 = {agent1->row, agent1->col};
          pos2 = {agent2->row, agent2->col};
        } else if (agent1_moved) {
          // Only agent1 moved - agent1 bounces back
          pos1 = {agent1->row, agent1->col};
        } else if (agent2_moved) {
          // Only agent2 moved - agent2 bounces back
          pos2 = {agent2->row, agent2->col};
        }
        // If neither moved, they both stay (already correct)
      }
    }
  }
}

void Grid::ApplyMoves(const std::vector<std::pair<int, std::pair<int, int>>>& moves) {
  for (const auto& move : moves) {
    int agent_id = move.first;
    int new_row = move.second.first;
    int new_col = move.second.second;

    AgentData* agent = GetAgent(agent_id);
    if (agent != nullptr) {
      agent->row = new_row;
      agent->col = new_col;
    }
  }
}

void Grid::ProcessInteractions(const std::vector<ActionType>& actions) {
  for (size_t i = 0; i < agents_.size() && i < actions.size(); ++i) {
    if (actions[i] != ActionType::kInteract) continue;

    AgentData& agent = agents_[i];

    // Calculate the position the agent is facing
    auto [front_row, front_col] = GetNextPosition(agent.row, agent.col, agent.direction);

    if (!IsWithinBounds(front_row, front_col)) continue;

    // Try to interact with a door
    DoorData* door = GetDoorAt(front_row, front_col);
    if (door != nullptr) {
      if (!door->is_open && agent.HasKey(door->required_key_color)) {
        // Open the door and consume the key
        door->is_open = true;
        agent.RemoveKey(door->required_key_color);
      } else if (door->is_open) {
        // Close the door
        door->is_open = false;
      }
      continue;
    }

    // Try to pick up ground items at agent's current position
    auto items = GetGroundItemsAt(agent.row, agent.col);
    for (const auto& ground_item : items) {
      if (agent.inventory.size() < kMaxInventorySize) {
        agent.inventory.push_back(ground_item.item);
        RemoveGroundItem(agent.row, agent.col, ground_item.item.id);
        break; // Pick up one item at a time
      }
    }
  }
}

std::pair<int, int> Grid::FindEmptyCell() const {
  for (int row = 0; row < rows_; ++row) {
    for (int col = 0; col < cols_; ++col) {
      if (GetCell(row, col) == CellType::kEmpty &&
          GetAgentAt(row, col) == nullptr &&
          GetDoorAt(row, col) == nullptr &&
          GetGroundItemsAt(row, col).empty()) {
        return {row, col};
      }
    }
  }
  return {-1, -1}; // No empty cell found
}

std::vector<std::pair<int, int>> Grid::FindEmptyCells() const {
  std::vector<std::pair<int, int>> empty_cells;
  for (int row = 0; row < rows_; ++row) {
    for (int col = 0; col < cols_; ++col) {
      if (GetCell(row, col) == CellType::kEmpty &&
          GetAgentAt(row, col) == nullptr &&
          GetDoorAt(row, col) == nullptr &&
          GetGroundItemsAt(row, col).empty()) {
        empty_cells.emplace_back(row, col);
      }
    }
  }
  return empty_cells;
}

std::vector<int> Grid::GetDeadAgents() const {
  std::vector<int> dead_agents;
  for (const auto& agent : agents_) {
    if (GetCell(agent.row, agent.col) == CellType::kLava) {
      dead_agents.push_back(agent.id);
    }
  }
  return dead_agents;
}

void Grid::RemoveDeadAgents() {
  agents_.erase(
      std::remove_if(agents_.begin(), agents_.end(),
                     [this](const AgentData& agent) {
                       return GetCell(agent.row, agent.col) == CellType::kLava;
                     }),
      agents_.end());
}

std::string Grid::ToString() const {
  std::ostringstream oss;

  for (int row = 0; row < rows_; ++row) {
    for (int col = 0; col < cols_; ++col) {
      // Check for agents first (they are on top)
      const AgentData* agent = GetAgentAt(row, col);
      if (agent != nullptr) {
        // Use direction symbols for agents
        switch (agent->direction) {
          case Direction::kNorth: oss << '^'; break;
          case Direction::kEast:  oss << '>'; break;
          case Direction::kSouth: oss << 'v'; break;
          case Direction::kWest:  oss << '<'; break;
        }
      } else {
        // Check for doors
        const DoorData* door = GetDoorAt(row, col);
        if (door != nullptr) {
          oss << (door->is_open ? '/' : '+');
        } else {
          // Check for ground items
          auto items = GetGroundItemsAt(row, col);
          if (!items.empty()) {
            oss << '*';
          } else {
            // Show cell type
            switch (GetCell(row, col)) {
              case CellType::kEmpty:   oss << '.'; break;
              case CellType::kWall:    oss << '#'; break;
              case CellType::kLava:    oss << '~'; break;
              case CellType::kGoal:    oss << 'G'; break;
              case CellType::kSynchro: oss << 'S'; break;
            }
          }
        }
      }
    }
    oss << '\n';
  }

  return oss.str();
}

int Grid::FindAgentIndex(int agent_id) const {
  for (size_t i = 0; i < agents_.size(); ++i) {
    if (agents_[i].id == agent_id) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

bool Grid::HasNonOverlappableActor(int row, int col) const {
  // Check for agents
  if (GetAgentAt(row, col) != nullptr) {
    return true;
  }

  // Check for closed doors
  const DoorData* door = GetDoorAt(row, col);
  if (door != nullptr && !door->is_open) {
    return true;
  }

  return false;
}

}  // namespace companion
}  // namespace open_spiel