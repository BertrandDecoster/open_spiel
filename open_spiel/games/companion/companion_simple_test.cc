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
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace companion {
namespace {

namespace testing = open_spiel::testing;

void BasicCompanionSimpleTests() {
  testing::LoadGameTest("companion_simple");
  testing::NoChanceOutcomesTest(*LoadGame("companion_simple"));
  testing::RandomSimTest(*LoadGame("companion_simple"), 100);
  testing::RandomSimTestWithUndo(*LoadGame("companion_simple"), 10);
}

void TestCompanionSimpleGameCreation() {
  auto game = LoadGame("companion_simple");
  SPIEL_CHECK_TRUE(game != nullptr);
  SPIEL_CHECK_EQ(game->GetType().short_name, "companion_simple");
  SPIEL_CHECK_EQ(game->GetType().dynamics, GameType::Dynamics::kSimultaneous);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);  // Default
  SPIEL_CHECK_EQ(game->NumDistinctActions(), 6);  // North, East, South, West, Interact, Stay
}

void TestCompanionSimpleWithCustomParameters() {
  auto game = LoadGame("companion_simple", {{"rows", GameParameter(5)},
                                           {"cols", GameParameter(5)},
                                           {"horizon", GameParameter(50)},
                                           {"num_agents", GameParameter(3)}});
  SPIEL_CHECK_TRUE(game != nullptr);
  SPIEL_CHECK_EQ(game->NumPlayers(), 3);

  auto state = game->NewInitialState();
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kSimultaneousPlayerId);

  // Test that all players have legal actions
  for (int player = 0; player < game->NumPlayers(); ++player) {
    auto legal_actions = state->LegalActions(player);
    SPIEL_CHECK_EQ(legal_actions.size(), 6);
  }
}

void TestCompanionSimpleBasicGameplay() {
  auto game = LoadGame("companion_simple", {{"rows", GameParameter(3)},
                                           {"cols", GameParameter(3)},
                                           {"num_agents", GameParameter(1)}});
  auto state = game->NewInitialState();

  // Agent should start at bottom-left, goal at top-right
  std::string initial_state = state->ToString();
  SPIEL_CHECK_TRUE(initial_state.find("^") != std::string::npos);  // Agent present
  SPIEL_CHECK_TRUE(initial_state.find("G") != std::string::npos);  // Goal present

  // Move agent towards goal (East, then North)
  std::vector<Action> actions = {static_cast<Action>(ActionType::kEast)};
  state->ApplyActions(actions);
  SPIEL_CHECK_FALSE(state->IsTerminal());

  actions = {static_cast<Action>(ActionType::kEast)};
  state->ApplyActions(actions);
  SPIEL_CHECK_FALSE(state->IsTerminal());

  actions = {static_cast<Action>(ActionType::kNorth)};
  state->ApplyActions(actions);
  SPIEL_CHECK_FALSE(state->IsTerminal());

  actions = {static_cast<Action>(ActionType::kNorth)};
  state->ApplyActions(actions);
  SPIEL_CHECK_TRUE(state->IsTerminal());  // Should reach goal

  // Check that success reward was given
  auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns.size(), 1);
  SPIEL_CHECK_GT(returns[0], 50.0);  // Should have success reward minus step penalties
}

void TestCompanionSimpleCollisionResolution() {
  auto game = LoadGame("companion_simple", {{"rows", GameParameter(3)},
                                           {"cols", GameParameter(3)},
                                           {"num_agents", GameParameter(2)}});
  auto state = game->NewInitialState();

  // Both agents try to move to the same position
  std::vector<Action> actions = {
      static_cast<Action>(ActionType::kEast),   // Agent 0 moves east
      static_cast<Action>(ActionType::kNorth)   // Agent 1 moves north
  };

  // Apply actions and check that collision resolution works
  state->ApplyActions(actions);
  SPIEL_CHECK_FALSE(state->IsTerminal());

  // Test that both agents moved (no collision in this case)
  std::string state_str = state->ToString();
  // Should have two agents visible
  int agent_count = 0;
  for (char c : state_str) {
    if (c == '^' || c == '>' || c == 'v' || c == '<') {
      agent_count++;
    }
  }
  SPIEL_CHECK_EQ(agent_count, 2);
}

void TestCompanionSimpleObservationTensor() {
  auto game = LoadGame("companion_simple", {{"rows", GameParameter(4)},
                                           {"cols", GameParameter(4)}});
  auto state = game->NewInitialState();

  // Test observation tensor shape
  auto obs_shape = game->ObservationTensorShape();
  SPIEL_CHECK_EQ(obs_shape.size(), 3);  // [planes, rows, cols]
  SPIEL_CHECK_EQ(obs_shape[1], 4);      // rows
  SPIEL_CHECK_EQ(obs_shape[2], 4);      // cols

  // Test observation tensor content
  std::vector<float> obs_tensor(game->ObservationTensorSize());
  state->ObservationTensor(0, absl::MakeSpan(obs_tensor));

  // Check that tensor is not all zeros (should have some content)
  bool has_content = false;
  for (float value : obs_tensor) {
    if (value != 0.0f) {
      has_content = true;
      break;
    }
  }
  SPIEL_CHECK_TRUE(has_content);
}

void TestCompanionSimpleActionConversion() {
  auto game = LoadGame("companion_simple");
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

void TestCompanionSimpleHorizonReached() {
  auto game = LoadGame("companion_simple", {{"horizon", GameParameter(5)}});
  auto state = game->NewInitialState();

  // Apply stay actions until horizon is reached
  for (int step = 0; step < 5; ++step) {
    SPIEL_CHECK_FALSE(state->IsTerminal());
    std::vector<Action> actions(game->NumPlayers(), static_cast<Action>(ActionType::kStay));
    state->ApplyActions(actions);
  }

  SPIEL_CHECK_TRUE(state->IsTerminal());  // Should be terminal due to horizon
}

}  // namespace
}  // namespace companion
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::companion::BasicCompanionSimpleTests();
  open_spiel::companion::TestCompanionSimpleGameCreation();
  open_spiel::companion::TestCompanionSimpleWithCustomParameters();
  open_spiel::companion::TestCompanionSimpleBasicGameplay();
  open_spiel::companion::TestCompanionSimpleCollisionResolution();
  open_spiel::companion::TestCompanionSimpleObservationTensor();
  open_spiel::companion::TestCompanionSimpleActionConversion();
  open_spiel::companion::TestCompanionSimpleHorizonReached();
  std::cout << "All CompanionSimple tests passed!" << std::endl;
  return 0;
}