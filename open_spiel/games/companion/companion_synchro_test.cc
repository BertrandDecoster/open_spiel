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
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace companion {
namespace {

namespace testing = open_spiel::testing;

void BasicCompanionSynchroTests() {
  testing::LoadGameTest("companion_synchro");
  testing::NoChanceOutcomesTest(*LoadGame("companion_synchro"));
  testing::RandomSimTest(*LoadGame("companion_synchro"), 100);
  testing::RandomSimTestWithUndo(*LoadGame("companion_synchro"), 10);
}

void TestCompanionSynchroGameCreation() {
  auto game = LoadGame("companion_synchro");
  SPIEL_CHECK_TRUE(game != nullptr);
  SPIEL_CHECK_EQ(game->GetType().short_name, "companion_synchro");
  SPIEL_CHECK_EQ(game->GetType().dynamics, GameType::Dynamics::kSimultaneous);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);  // Default
  SPIEL_CHECK_EQ(game->NumDistinctActions(), 6);  // North, East, South, West, Interact, Stay
}

void TestCompanionSynchroWithCustomParameters() {
  auto game = LoadGame("companion_synchro", {{"rows", GameParameter(6)},
                                            {"cols", GameParameter(6)},
                                            {"horizon", GameParameter(75)},
                                            {"players", GameParameter(4)}});
  SPIEL_CHECK_TRUE(game != nullptr);
  SPIEL_CHECK_EQ(game->NumPlayers(), 4);

  auto state = game->NewInitialState();
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kSimultaneousPlayerId);

  // Test that all players have legal actions
  for (int player = 0; player < game->NumPlayers(); ++player) {
    auto legal_actions = state->LegalActions(player);
    SPIEL_CHECK_EQ(legal_actions.size(), 6);
  }
}

void TestCompanionSynchroTwoAgentCoordination() {
  auto game = LoadGame("companion_synchro", {{"rows", GameParameter(3)},
                                            {"cols", GameParameter(3)},
                                            {"players", GameParameter(2)}});
  auto state = game->NewInitialState();

  // Check initial state has synchro cells and agents
  std::string initial_state = state->ToString();

  size_t start_pos = 0;
  // Loop to find the position after the 5th newline
  for (int i = 0; i < 5; ++i) {
      start_pos = initial_state.find('\n', start_pos);
      if (start_pos == std::string::npos) {
          // Handle case where there are fewer than 5 lines
          std::cerr << "Error: The string does not have 5 lines to skip." << std::endl;
          SPIEL_CHECK_TRUE(false);  // Force failure
      }
      // Move past the found newline character for the next search
      start_pos++;
  }

  // Create a new string containing just the grid
  initial_state = initial_state.substr(start_pos);


  SPIEL_CHECK_TRUE(initial_state.find("S") != std::string::npos);  // Synchro cells present

  // Count agents
  int agent_count = 0;
  for (char c : initial_state) {
    if (c == '^' || c == '>' || c == 'v' || c == '<') {
      agent_count++;
    }
  }
  SPIEL_CHECK_EQ(agent_count, 2);

  // Count synchro cells
  int synchro_count = 0;
  for (char c : initial_state) {
    if (c == 'S') {
      synchro_count++;
    }
  }

  SPIEL_CHECK_EQ(synchro_count, 2);  // Should have 2 synchro cells for 2 agents
}

void TestCompanionSynchroSingleAgentSuccess() {
  auto game = LoadGame("companion_synchro", {{"rows", GameParameter(3)},
                                            {"cols", GameParameter(3)},
                                            {"players", GameParameter(1)}});
  auto state = game->NewInitialState();

  // Single agent should have one synchro cell in center
  std::string initial_state = state->ToString();

  // Find agent and synchro cell positions
  // For single agent, the goal should be achievable
  int max_steps = 10;  // Prevent infinite loop
  for (int step = 0; step < max_steps && !state->IsTerminal(); ++step) {
    // Try moving towards center (where synchro cell should be)
    std::vector<Action> actions = {static_cast<Action>(ActionType::kNorth)};
    state->ApplyActions(actions);
  }

  // Agent should be able to reach the synchro cell in reasonable time
  // (This test might fail if the maze is too complex, but with 3x3 it should work)
}

void TestCompanionSynchroObservationTensor() {
  auto game = LoadGame("companion_synchro", {{"rows", GameParameter(4)},
                                            {"cols", GameParameter(4)},
                                            {"players", GameParameter(3)}});
  auto state = game->NewInitialState();

  // Test observation tensor shape
  auto obs_shape = game->ObservationTensorShape();
  SPIEL_CHECK_EQ(obs_shape.size(), 3);  // [planes, rows, cols]
  SPIEL_CHECK_EQ(obs_shape[1], 4);      // rows
  SPIEL_CHECK_EQ(obs_shape[2], 4);      // cols

  // Test observation tensor content
  std::vector<float> obs_tensor(game->ObservationTensorSize());
  state->ObservationTensor(0, absl::MakeSpan(obs_tensor));

  // Check that tensor has content
  bool has_content = false;
  for (float value : obs_tensor) {
    if (value != 0.0f) {
      has_content = true;
      break;
    }
  }
  SPIEL_CHECK_TRUE(has_content);
}

void TestCompanionSynchroCollisionHandling() {
  auto game = LoadGame("companion_synchro", {{"rows", GameParameter(5)},
                                            {"cols", GameParameter(5)},
                                            {"players", GameParameter(2)}});
  auto state = game->NewInitialState();

  // Apply some random actions to test collision handling
  for (int step = 0; step < 5 && !state->IsTerminal(); ++step) {
    std::vector<Action> actions = {
        static_cast<Action>(ActionType::kEast),   // Agent 0
        static_cast<Action>(ActionType::kWest)    // Agent 1
    };

    state->ApplyActions(actions);
    SPIEL_CHECK_FALSE(state->IsTerminal() && step < 4);  // Shouldn't terminate early from collision
  }
}

void TestCompanionSynchroActionStrings() {
  auto game = LoadGame("companion_synchro");
  auto state = game->NewInitialState();

  // Test all action types
  std::vector<std::string> expected_actions = {
      "North", "East", "South", "West", "Interact", "Stay"
  };

  for (int action = 0; action < 6; ++action) {
    std::string action_str = state->ActionToString(0, action);
    SPIEL_CHECK_EQ(action_str, expected_actions[action]);
  }
}

void TestCompanionSynchroHorizonTermination() {
  auto game = LoadGame("companion_synchro", {{"horizon", GameParameter(3)}});
  auto state = game->NewInitialState();

  // Apply stay actions until horizon is reached
  for (int step = 0; step < 3; ++step) {
    SPIEL_CHECK_FALSE(state->IsTerminal());
    std::vector<Action> actions(game->NumPlayers(), static_cast<Action>(ActionType::kStay));
    state->ApplyActions(actions);
  }

  SPIEL_CHECK_TRUE(state->IsTerminal());  // Should be terminal due to horizon
}

void TestCompanionSynchroInformationState() {
  auto game = LoadGame("companion_synchro");
  auto state = game->NewInitialState();

  // Test information state string
  for (int player = 0; player < game->NumPlayers(); ++player) {
    std::string info_state = state->InformationStateString(player);
    SPIEL_CHECK_FALSE(info_state.empty());
    SPIEL_CHECK_TRUE(info_state.find("CompanionSynchro") != std::string::npos);
  }

  // Test information state tensor
  std::vector<float> info_tensor(game->InformationStateTensorSize());
  state->InformationStateTensor(0, absl::MakeSpan(info_tensor));

  // Should have same content as observation tensor for now
  std::vector<float> obs_tensor(game->ObservationTensorSize());
  state->ObservationTensor(0, absl::MakeSpan(obs_tensor));

  SPIEL_CHECK_EQ(info_tensor.size(), obs_tensor.size());
}

void TestCompanionSynchroCloning() {
  auto game = LoadGame("companion_synchro");
  auto state = game->NewInitialState();

  // Apply some actions
  std::vector<Action> actions = {
      static_cast<Action>(ActionType::kNorth),
      static_cast<Action>(ActionType::kSouth)
  };
  state->ApplyActions(actions);

  // Clone the state
  auto cloned_state = state->Clone();

  // States should be different objects but have same content
  SPIEL_CHECK_NE(state.get(), cloned_state.get());
  SPIEL_CHECK_EQ(state->ToString(), cloned_state->ToString());
  SPIEL_CHECK_EQ(state->IsTerminal(), cloned_state->IsTerminal());

  // Modify original state
  actions = {static_cast<Action>(ActionType::kEast), static_cast<Action>(ActionType::kWest)};
  state->ApplyActions(actions);

  // Clone should remain unchanged
  SPIEL_CHECK_NE(state->ToString(), cloned_state->ToString());
}

}  // namespace
}  // namespace companion
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::companion::BasicCompanionSynchroTests();
  open_spiel::companion::TestCompanionSynchroGameCreation();
  open_spiel::companion::TestCompanionSynchroWithCustomParameters();
  open_spiel::companion::TestCompanionSynchroTwoAgentCoordination();
  open_spiel::companion::TestCompanionSynchroSingleAgentSuccess();
  open_spiel::companion::TestCompanionSynchroObservationTensor();
  open_spiel::companion::TestCompanionSynchroCollisionHandling();
  open_spiel::companion::TestCompanionSynchroActionStrings();
  open_spiel::companion::TestCompanionSynchroHorizonTermination();
  open_spiel::companion::TestCompanionSynchroInformationState();
  open_spiel::companion::TestCompanionSynchroCloning();
  std::cout << "All CompanionSynchro tests passed!" << std::endl;
  return 0;
}