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

#ifndef OPEN_SPIEL_GAMES_COMPANION_BASE_H_
#define OPEN_SPIEL_GAMES_COMPANION_BASE_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/games/companion/companion_grid.h"
#include "open_spiel/games/companion/companion_types.h"

namespace open_spiel {
namespace companion {

class CompanionGame;

// Base class for all Companion environment states
class CompanionState : public SimMoveState {
 public:
  CompanionState(std::shared_ptr<const Game> game, int horizon, int num_agents);
  CompanionState(const CompanionState& other) = default;
  virtual ~CompanionState() = default;

  // OpenSpiel State interface
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player, absl::Span<float> values) const override;
  void InformationStateTensor(Player player, absl::Span<float> values) const override;
  // Clone() is implemented by derived classes since CompanionState is abstract
  std::vector<Action> LegalActions(Player player) const override;

 protected:
  // Core game loop implementation
  void DoApplyActions(const std::vector<Action>& actions) override;

  // Virtual methods for child classes to implement
  virtual void SetupGrid() = 0;
  virtual bool CheckTermination() const = 0;
  virtual std::vector<double> ComputeRewards() const = 0;
  virtual std::string GetEnvironmentName() const = 0;

  // Helper methods for child classes
  void AddAgent(int id, int row, int col, Direction dir, Color color, AgentType type);
  void PlaceGoal(int row, int col);
  void PlaceSynchroCell(int row, int col);
  void PlaceWall(int row, int col);
  void PlaceLava(int row, int col);

  // Observation tensor helpers
  virtual void WriteObservationTensor(Player player, absl::Span<float> values) const;
  int GetObservationTensorSize() const;

 public:
  // Constants (public so derived games can access them)
  static constexpr double kStepReward = -1.0;
  static constexpr double kSuccessReward = 100.0;
  static constexpr double kDeathPenalty = -100.0;

 protected:
  // Game state
  Grid grid_;
  int horizon_;
  int num_agents_;
  int timestep_;
  std::vector<double> rewards_;
  std::vector<double> returns_;
  bool is_terminal_;

 private:
  // Process one timestep
  void ProcessTimestep(const std::vector<ActionType>& typed_actions);

  // Convert Action to ActionType
  ActionType ActionToActionType(Action action) const;

  // Update FSM agents (if any)
  void UpdateFSMAgents();

  // Helper for calculating observation dimensions
  int GetNumObservationPlanes() const;
};

// Base class for all Companion games
class CompanionGame : public SimMoveGame {
 public:
  explicit CompanionGame(const GameParameters& params, const GameType& game_type);
  virtual ~CompanionGame() = default;

  // OpenSpiel Game interface
  int NumDistinctActions() const override { return kNumActions; }
  int NumPlayers() const override { return num_agents_; }
  double MinUtility() const override;
  double MaxUtility() const override;
  std::vector<int> ObservationTensorShape() const override;
  std::vector<int> InformationStateTensorShape() const override;
  int MaxGameLength() const override { return horizon_; }

  // Getters for child classes
  int GetRows() const { return rows_; }
  int GetCols() const { return cols_; }
  int GetHorizon() const { return horizon_; }
  int GetNumAgents() const { return num_agents_; }

 protected:
  int rows_;
  int cols_;
  int horizon_;
  int num_agents_;

 private:
  // Calculate observation tensor dimensions
  int GetNumObservationPlanes() const;
};

}  // namespace companion
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_COMPANION_BASE_H_