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

#include "open_spiel/games/companion/companion_synchro.h"

#include <memory>
#include <set>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace companion {
namespace {

// Default parameters for synchro game
constexpr int kSynchroDefaultRows = 8;
constexpr int kSynchroDefaultCols = 8;
constexpr int kSynchroDefaultHorizon = 100;
constexpr int kSynchroDefaultNumAgents = 2;

// Game type
const GameType kSynchroGameType = {
    /*short_name=*/"companion_synchro",
    /*long_name=*/"Companion Synchro Environment",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kIdentical,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/kMaxAgents,
    /*min_num_players=*/1,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"rows", GameParameter(kSynchroDefaultRows)},
     {"cols", GameParameter(kSynchroDefaultCols)},
     {"horizon", GameParameter(kSynchroDefaultHorizon)},
     {"players", GameParameter(kSynchroDefaultNumAgents)}}
};

std::shared_ptr<const Game> SynchroFactory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CompanionSynchroGame(params));
}

REGISTER_SPIEL_GAME(kSynchroGameType, SynchroFactory);

}  // namespace

using namespace companion;

CompanionSynchroState::CompanionSynchroState(std::shared_ptr<const Game> game,
                                           int horizon, int num_agents)
    : CompanionState(game, horizon, num_agents) {
  synchro_positions_.reserve(num_agents);
  SetupGrid();
}

std::unique_ptr<State> CompanionSynchroState::Clone() const {
  return std::unique_ptr<State>(new CompanionSynchroState(*this));
}

void CompanionSynchroState::SetupGrid() {
  const CompanionSynchroGame* game = static_cast<const CompanionSynchroGame*>(GetGame().get());
  const int rows = game->GetRows();
  const int cols = game->GetCols();

  // Place synchro cells in a pattern that requires coordination
  synchro_positions_.clear();

  if (num_agents_ == 1) {
    // Single agent case - place synchro cell in center
    int center_row = rows / 2;
    int center_col = cols / 2;
    synchro_positions_.emplace_back(center_row, center_col);
    PlaceSynchroCell(center_row, center_col);
  } else if (num_agents_ == 2) {
    // Two agents - place synchro cells at opposite corners
    synchro_positions_.emplace_back(0, 0);                    // Top-left
    synchro_positions_.emplace_back(rows - 1, cols - 1);      // Bottom-right
    PlaceSynchroCell(0, 0);
    PlaceSynchroCell(rows - 1, cols - 1);
  } else if (num_agents_ == 3) {
    // Three agents - triangle formation
    synchro_positions_.emplace_back(0, cols / 2);             // Top-center
    synchro_positions_.emplace_back(rows - 1, 0);             // Bottom-left
    synchro_positions_.emplace_back(rows - 1, cols - 1);      // Bottom-right
    PlaceSynchroCell(0, cols / 2);
    PlaceSynchroCell(rows - 1, 0);
    PlaceSynchroCell(rows - 1, cols - 1);
  } else if (num_agents_ == 4) {
    // Four agents - corners
    synchro_positions_.emplace_back(0, 0);                    // Top-left
    synchro_positions_.emplace_back(0, cols - 1);             // Top-right
    synchro_positions_.emplace_back(rows - 1, 0);             // Bottom-left
    synchro_positions_.emplace_back(rows - 1, cols - 1);      // Bottom-right
    PlaceSynchroCell(0, 0);
    PlaceSynchroCell(0, cols - 1);
    PlaceSynchroCell(rows - 1, 0);
    PlaceSynchroCell(rows - 1, cols - 1);
  } else {
    // More agents - distribute around the perimeter
    for (int i = 0; i < num_agents_; ++i) {
      int row, col;
      if (i < cols) {
        // Top row
        row = 0;
        col = i;
      } else if (i < cols + rows - 1) {
        // Right column
        row = i - cols + 1;
        col = cols - 1;
      } else if (i < 2 * cols + rows - 2) {
        // Bottom row (right to left)
        row = rows - 1;
        col = cols - 1 - (i - cols - rows + 1);
      } else {
        // Left column (bottom to top)
        row = rows - 1 - (i - 2 * cols - rows + 2);
        col = 0;
      }

      // Ensure we don't exceed bounds
      row = std::max(0, std::min(row, rows - 1));
      col = std::max(0, std::min(col, cols - 1));

      synchro_positions_.emplace_back(row, col);
      PlaceSynchroCell(row, col);
    }
  }

  // Place agents at the center of the grid
  std::vector<Color> agent_colors = {
      Color::kRed, Color::kBlue, Color::kGreen, Color::kYellow,
      Color::kPurple, Color::kOrange, Color::kCyan, Color::kPink
  };

  int center_row = rows / 2;
  int center_col = cols / 2;

  for (int i = 0; i < num_agents_; ++i) {
    // Spread agents around the center
    int offset_row = (i % 2 == 0) ? 0 : ((i % 4 < 2) ? -1 : 1);
    int offset_col = ((i / 2) % 2 == 0) ? 0 : ((i / 4 < 2) ? -1 : 1);

    int agent_row = std::max(0, std::min(center_row + offset_row, rows - 1));
    int agent_col = std::max(0, std::min(center_col + offset_col, cols - 1));

    Color color = agent_colors[i % agent_colors.size()];
    AddAgent(i, agent_row, agent_col, Direction::kNorth, color, AgentType::kRL);
  }

  // Add some walls to create interesting paths
  if (rows >= 6 && cols >= 6) {
    // Create walls that require coordination to navigate around

    // Central cross pattern
    int mid_row = rows / 2;
    int mid_col = cols / 2;

    // Horizontal walls
    for (int col = 1; col < cols - 1; ++col) {
      if (col != mid_col) {  // Leave gap in the middle
        PlaceWall(mid_row - 1, col);
        PlaceWall(mid_row + 1, col);
      }
    }

    // Vertical walls
    for (int row = 1; row < rows - 1; ++row) {
      if (row != mid_row) {  // Leave gap in the middle
        PlaceWall(row, mid_col - 1);
        PlaceWall(row, mid_col + 1);
      }
    }
  }
}

bool CompanionSynchroState::CheckTermination() const {
  // Check if all synchro cells are occupied
  std::set<std::pair<int, int>> occupied_synchro_positions;

  for (const auto& agent : grid_.GetAgents()) {
    for (const auto& synchro_pos : synchro_positions_) {
      if (agent.row == synchro_pos.first && agent.col == synchro_pos.second) {
        occupied_synchro_positions.insert(synchro_pos);
        break; // Agent can only occupy one synchro cell
      }
    }
  }

  // Success if all synchro positions are occupied
  return occupied_synchro_positions.size() == synchro_positions_.size();
}

std::vector<double> CompanionSynchroState::ComputeRewards() const {
  std::vector<double> rewards(num_agents_, 0.0);

  // If terminated due to synchronization achieved, all agents get success reward
  if (CheckTermination()) {
    std::fill(rewards.begin(), rewards.end(), kSuccessReward);
  }

  return rewards;
}

CompanionSynchroGame::CompanionSynchroGame(const GameParameters& params)
    : CompanionGame(params, kSynchroGameType) {}

std::unique_ptr<State> CompanionSynchroGame::NewInitialState() const {
  return std::unique_ptr<State>(
      new CompanionSynchroState(shared_from_this(), horizon_, num_agents_));
}


}  // namespace companion
}  // namespace open_spiel