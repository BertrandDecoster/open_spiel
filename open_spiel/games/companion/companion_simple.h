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

#ifndef OPEN_SPIEL_GAMES_COMPANION_SIMPLE_H_
#define OPEN_SPIEL_GAMES_COMPANION_SIMPLE_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/companion/companion_base.h"

namespace open_spiel {
namespace companion {

// Simple Companion environment where agents try to reach a single goal cell.
// The game ends when any agent reaches the goal, and all agents receive
// a positive reward.
//
// Parameters:
//   "rows"       int     Number of rows in the grid           (default = 8)
//   "cols"       int     Number of columns in the grid        (default = 8)
//   "horizon"    int     Maximum episode length               (default = 100)
//   "num_agents" int     Number of agents                     (default = 2)

class CompanionSimpleState : public CompanionState {
 public:
  CompanionSimpleState(std::shared_ptr<const Game> game, int horizon, int num_agents);
  CompanionSimpleState(const CompanionSimpleState& other) = default;

  std::unique_ptr<State> Clone() const override;

 protected:
  void SetupGrid() override;
  bool CheckTermination() const override;
  std::vector<double> ComputeRewards() const override;
  std::string GetEnvironmentName() const override { return "CompanionSimple"; }

 private:
  // Track goal position for efficient termination checking
  int goal_row_, goal_col_;
};

class CompanionSimpleGame : public CompanionGame {
 public:
  explicit CompanionSimpleGame(const GameParameters& params);

  std::unique_ptr<State> NewInitialState() const override;

  // Game information
  std::string ToString() const;

 private:
  static GameType CreateGameType();
};

}  // namespace companion
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_COMPANION_SIMPLE_H_