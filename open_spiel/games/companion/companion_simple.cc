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

#include "open_spiel/games/companion/companion_simple.h"

#include <memory>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace companion {
namespace {

// Default parameters
constexpr int kDefaultRows = 8;
constexpr int kDefaultCols = 8;
constexpr int kDefaultHorizon = 100;
constexpr int kDefaultNumAgents = 2;

// Game type
const GameType kSimpleGameType = {
    /*short_name=*/"companion_simple",
    /*long_name=*/"Companion Simple Environment",
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
    {{"rows", GameParameter(kDefaultRows)},
     {"cols", GameParameter(kDefaultCols)},
     {"horizon", GameParameter(kDefaultHorizon)},
     {"num_agents", GameParameter(kDefaultNumAgents)}}
};

std::shared_ptr<const Game> SimpleFactory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CompanionSimpleGame(params));
}

REGISTER_SPIEL_GAME(kSimpleGameType, SimpleFactory);

}  // namespace

using namespace companion;

CompanionSimpleState::CompanionSimpleState(std::shared_ptr<const Game> game,
                                         int horizon, int num_agents)
    : CompanionState(game, horizon, num_agents), goal_row_(-1), goal_col_(-1) {
  SetupGrid();
}

std::unique_ptr<State> CompanionSimpleState::Clone() const {
  return std::unique_ptr<State>(new CompanionSimpleState(*this));
}

void CompanionSimpleState::SetupGrid() {
  const CompanionSimpleGame* game = static_cast<const CompanionSimpleGame*>(GetGame().get());
  const int rows = game->GetRows();
  const int cols = game->GetCols();

  // Place goal in the top-right corner
  goal_row_ = 0;
  goal_col_ = cols - 1;
  PlaceGoal(goal_row_, goal_col_);

  // Place agents in bottom-left area
  std::vector<Color> agent_colors = {
      Color::kRed, Color::kBlue, Color::kGreen, Color::kYellow,
      Color::kPurple, Color::kOrange, Color::kCyan, Color::kPink
  };

  for (int i = 0; i < num_agents_; ++i) {
    int agent_row = rows - 1;
    int agent_col = i % cols;  // Spread agents across bottom row

    // Ensure we don't exceed grid bounds
    if (agent_col >= cols) {
      agent_col = cols - 1;
      agent_row = std::max(0, rows - 2 - (i / cols));
    }

    Color color = agent_colors[i % agent_colors.size()];
    AddAgent(i, agent_row, agent_col, Direction::kNorth, color, AgentType::kRL);
  }

  // Add some walls to make the environment more interesting
  // Create a simple maze-like structure
  if (rows >= 5 && cols >= 5) {
    // Horizontal wall
    for (int col = 1; col < cols - 1; ++col) {
      if (col != cols / 2) {  // Leave a gap in the middle
        // Don't place walls on goal position
        if (!(rows / 2 == goal_row_ && col == goal_col_)) {
          PlaceWall(rows / 2, col);
        }
      }
    }

    // Vertical wall
    for (int row = 1; row < rows / 2; ++row) {
      // Don't place walls on goal position
      if (!(row == goal_row_ && cols / 2 == goal_col_)) {
        PlaceWall(row, cols / 2);
      }
    }
  }
}

bool CompanionSimpleState::CheckTermination() const {
  // Check if any agent has reached the goal
  for (const auto& agent : grid_.GetAgents()) {
    if (agent.row == goal_row_ && agent.col == goal_col_) {
      return true;
    }
  }
  return false;
}

std::vector<double> CompanionSimpleState::ComputeRewards() const {
  std::vector<double> rewards(num_agents_, 0.0);

  // If terminated due to goal reached, all agents get success reward
  bool goal_reached = false;
  for (const auto& agent : grid_.GetAgents()) {
    if (agent.row == goal_row_ && agent.col == goal_col_) {
      goal_reached = true;
      break;
    }
  }

  if (goal_reached) {
    std::fill(rewards.begin(), rewards.end(), kSuccessReward);
  }

  return rewards;
}

CompanionSimpleGame::CompanionSimpleGame(const GameParameters& params)
    : CompanionGame(params, kSimpleGameType) {}

std::unique_ptr<State> CompanionSimpleGame::NewInitialState() const {
  return std::unique_ptr<State>(
      new CompanionSimpleState(shared_from_this(), horizon_, num_agents_));
}

std::string CompanionSimpleGame::ToString() const {
  return absl::StrCat(
      "companion_simple(rows=", rows_,
      ",cols=", cols_,
      ",horizon=", horizon_,
      ",num_agents=", num_agents_, ")");
}

}  // namespace companion
}  // namespace open_spiel