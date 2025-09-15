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

#include "open_spiel/games/companion/companion_base.h"

#include <algorithm>
#include <sstream>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace companion {

CompanionState::CompanionState(std::shared_ptr<const Game> game, int horizon, int num_agents)
    : SimMoveState(game),
      grid_(static_cast<const CompanionGame*>(game.get())->GetRows(),
            static_cast<const CompanionGame*>(game.get())->GetCols()),
      horizon_(horizon),
      num_agents_(num_agents),
      timestep_(0),
      rewards_(num_agents, 0.0),
      returns_(num_agents, 0.0),
      is_terminal_(false) {

  // Note: SetupGrid() is called by derived class constructors after their construction
}

Player CompanionState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : kSimultaneousPlayerId;
}

std::string CompanionState::ActionToString(Player player, Action action_id) const {
  if (action_id < 0 || action_id >= kNumActions) {
    return "Invalid action";
  }

  switch (static_cast<ActionType>(action_id)) {
    case ActionType::kNorth: return "North";
    case ActionType::kEast: return "East";
    case ActionType::kSouth: return "South";
    case ActionType::kWest: return "West";
    case ActionType::kInteract: return "Interact";
    case ActionType::kStay: return "Stay";
    default: return "Unknown";
  }
}

std::string CompanionState::ToString() const {
  std::ostringstream oss;
  oss << GetEnvironmentName() << " State:\n";
  oss << "Timestep: " << timestep_ << "/" << horizon_ << "\n";
  oss << "Terminal: " << (is_terminal_ ? "true" : "false") << "\n";
  oss << "Returns: [";
  for (size_t i = 0; i < returns_.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << returns_[i];
  }
  oss << "]\n\n";
  oss << grid_.ToString();
  return oss.str();
}

bool CompanionState::IsTerminal() const {
  return is_terminal_ || timestep_ >= horizon_;
}

std::vector<double> CompanionState::Returns() const {
  return returns_;
}

std::vector<double> CompanionState::Rewards() const {
  return rewards_;
}

std::string CompanionState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_agents_);

  // For now, all agents have full observability
  return ToString();
}

std::string CompanionState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_agents_);

  // For now, all agents have full observability
  return ToString();
}

void CompanionState::ObservationTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_agents_);

  WriteObservationTensor(player, values);
}

void CompanionState::InformationStateTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_agents_);

  // For now, information state equals observation
  ObservationTensor(player, values);
}

// Clone() is implemented by derived classes since CompanionState is abstract

std::vector<Action> CompanionState::LegalActions(Player player) const {
  if (IsTerminal()) {
    return {};
  }

  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_agents_);

  // All actions are always legal for all players
  std::vector<Action> legal_actions;
  for (int action = 0; action < kNumActions; ++action) {
    legal_actions.push_back(action);
  }
  return legal_actions;
}

void CompanionState::DoApplyActions(const std::vector<Action>& actions) {
  if (IsTerminal()) {
    return;
  }

  // Convert actions to typed actions
  std::vector<ActionType> typed_actions;
  typed_actions.reserve(num_agents_);

  for (int i = 0; i < num_agents_; ++i) {
    Action action = (i < actions.size()) ? actions[i] : static_cast<Action>(ActionType::kStay);
    typed_actions.push_back(ActionToActionType(action));
  }

  // Process the timestep
  ProcessTimestep(typed_actions);
}

void CompanionState::ProcessTimestep(const std::vector<ActionType>& typed_actions) {
  // Clear previous rewards
  std::fill(rewards_.begin(), rewards_.end(), kStepReward);

  // Update FSM agents
  UpdateFSMAgents();

  // Predict moves
  auto moves = grid_.PredictMoves(typed_actions);

  // Resolve collisions
  grid_.ResolveCollisions(moves);

  // Apply moves
  grid_.ApplyMoves(moves);

  // Process interactions
  grid_.ProcessInteractions(typed_actions);

  // Remove dead agents and apply death penalties
  auto dead_agents = grid_.GetDeadAgents();
  for (int agent_id : dead_agents) {
    if (agent_id < rewards_.size()) {
      rewards_[agent_id] = kDeathPenalty;
    }
  }
  grid_.RemoveDeadAgents();

  // Check termination
  is_terminal_ = CheckTermination();

  // Compute rewards (child class specific)
  if (is_terminal_) {
    auto termination_rewards = ComputeRewards();
    for (size_t i = 0; i < rewards_.size() && i < termination_rewards.size(); ++i) {
      rewards_[i] += termination_rewards[i];
    }
  }

  // Update returns
  for (size_t i = 0; i < returns_.size() && i < rewards_.size(); ++i) {
    returns_[i] += rewards_[i];
  }

  // Increment timestep
  ++timestep_;
}

ActionType CompanionState::ActionToActionType(Action action) const {
  if (action < 0 || action >= kNumActions) {
    return ActionType::kStay;
  }
  return static_cast<ActionType>(action);
}

void CompanionState::UpdateFSMAgents() {
  // TODO: Implement FSM agent logic
  // For now, FSM agents just stay in place
}

void CompanionState::AddAgent(int id, int row, int col, Direction dir, Color color, AgentType type) {
  AgentData agent(id, row, col, dir, color, type);
  grid_.AddAgent(agent);
}

void CompanionState::PlaceGoal(int row, int col) {
  grid_.SetCell(row, col, CellType::kGoal);
}

void CompanionState::PlaceSynchroCell(int row, int col) {
  grid_.SetCell(row, col, CellType::kSynchro);
}

void CompanionState::PlaceWall(int row, int col) {
  grid_.SetCell(row, col, CellType::kWall);
}

void CompanionState::PlaceLava(int row, int col) {
  grid_.SetCell(row, col, CellType::kLava);
}

void CompanionState::WriteObservationTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), GetObservationTensorSize());

  // Clear the tensor
  std::fill(values.begin(), values.end(), 0.0f);

  const int rows = grid_.rows();
  const int cols = grid_.cols();
  const int num_planes = GetNumObservationPlanes();

  // Helper to set value in tensor
  auto set_value = [&](int plane, int row, int col, float value) {
    int index = plane * rows * cols + row * cols + col;
    if (index < values.size()) {
      values[index] = value;
    }
  };

  int plane = 0;

  // Plane 0-4: Cell types (one-hot encoding)
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      CellType cell = grid_.GetCell(row, col);
      set_value(static_cast<int>(cell), row, col, 1.0f);
    }
  }
  plane = 5;

  // Plane 5: Agents
  for (const auto& agent : grid_.GetAgents()) {
    set_value(plane, agent.row, agent.col, 1.0f);
  }
  plane++;

  // Plane 6: Player agent (the observing player)
  const AgentData* player_agent = grid_.GetAgent(player);
  if (player_agent != nullptr) {
    set_value(plane, player_agent->row, player_agent->col, 1.0f);
  }
  plane++;

  // Plane 7: Doors (closed)
  for (const auto& door : grid_.GetDoors()) {
    if (!door.is_open) {
      set_value(plane, door.row, door.col, 1.0f);
    }
  }
  plane++;

  // Plane 8: Doors (open)
  for (const auto& door : grid_.GetDoors()) {
    if (door.is_open) {
      set_value(plane, door.row, door.col, 1.0f);
    }
  }
  plane++;

  // Plane 9: Ground items
  for (const auto& item : grid_.GetGroundItems()) {
    set_value(plane, item.row, item.col, 1.0f);
  }
}

int CompanionState::GetObservationTensorSize() const {
  return GetNumObservationPlanes() * grid_.rows() * grid_.cols();
}

int CompanionState::GetNumObservationPlanes() const {
  // 5 cell types + agents + player + closed doors + open doors + ground items
  return 10;
}

// CompanionGame implementation

CompanionGame::CompanionGame(const GameParameters& params, const GameType& game_type)
    : SimMoveGame(game_type, params) {

  rows_ = ParameterValue<int>("rows", 8);
  cols_ = ParameterValue<int>("cols", 8);
  horizon_ = ParameterValue<int>("horizon", 100);
  num_agents_ = ParameterValue<int>("num_agents", 2);

  SPIEL_CHECK_GT(rows_, 0);
  SPIEL_CHECK_GT(cols_, 0);
  SPIEL_CHECK_GT(horizon_, 0);
  SPIEL_CHECK_GT(num_agents_, 0);
  SPIEL_CHECK_LE(num_agents_, kMaxAgents);
}

double CompanionGame::MinUtility() const {
  // Worst case: death penalty + step penalties for full horizon
  return CompanionState::kDeathPenalty + CompanionState::kStepReward * horizon_;
}

double CompanionGame::MaxUtility() const {
  // Best case: success reward (step penalties are unavoidable)
  return CompanionState::kSuccessReward + CompanionState::kStepReward;
}

std::vector<int> CompanionGame::ObservationTensorShape() const {
  return {GetNumObservationPlanes(), rows_, cols_};
}

std::vector<int> CompanionGame::InformationStateTensorShape() const {
  // Same as observation for now
  return ObservationTensorShape();
}

int CompanionGame::GetNumObservationPlanes() const {
  // Match the number of planes in CompanionState
  return 10;
}

}  // namespace companion
}  // namespace open_spiel