#!/usr/bin/env python3
"""Human player script for cooperative box pushing game."""

import pyspiel

def get_human_action(state, player_id):
    """Get action from human player."""
    print(f"\nPlayer {player_id}'s turn:")
    legal_actions = state.legal_actions(player_id)
    print("Legal actions:")
    for action in legal_actions:
        action_str = state.action_to_string(player_id, action)
        print(f"  {action}: {action_str}")

    while True:
        try:
            action = int(input(f"Player {player_id}, enter your action: "))
            if action in legal_actions:
                return action
            else:
                print(f"Invalid action {action}. Choose from {legal_actions}")
        except ValueError:
            print("Please enter a number.")

def play_game():
    """Play the cooperative box pushing game with human players."""
    game = pyspiel.load_game('coop_box_pushing')
    state = game.new_initial_state()

    print("Welcome to Cooperative Box Pushing!")
    print("Goal: Both players work together to push boxes to target locations.")
    print(f"Game has {game.num_players()} players")
    print("\nInitial state:")
    print(state)

    move_count = 0

    while not state.is_terminal():
        move_count += 1
        print(f"\n=== Move {move_count} ===")
        print("Current state:")
        print(state)

        if state.is_chance_node():
            # Handle chance nodes (if any)
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            import numpy as np
            action = np.random.choice(action_list, p=prob_list)
            print(f"Chance outcome: {state.action_to_string(action)}")
            state.apply_action(action)
        elif state.is_simultaneous_node():
            # Both players choose actions simultaneously
            print("\nBoth players choose actions simultaneously!")
            chosen_actions = []
            for pid in range(game.num_players()):
                action = get_human_action(state, pid)
                chosen_actions.append(action)

            print(f"\nApplying actions: {[state.action_to_string(pid, action) for pid, action in enumerate(chosen_actions)]}")
            state.apply_actions(chosen_actions)
        else:
            # Sequential move
            current_player = state.current_player()
            action = get_human_action(state, current_player)
            action_str = state.action_to_string(current_player, action)
            print(f"Player {current_player} chose: {action_str}")
            state.apply_action(action)

    print("\n=== GAME OVER ===")
    print("Final state:")
    print(state)

    returns = state.returns()
    print(f"\nResults:")
    for pid in range(game.num_players()):
        print(f"Player {pid} utility: {returns[pid]}")

    total_reward = sum(returns)
    print(f"Total cooperative reward: {total_reward}")

    if total_reward > 0:
        print("🎉 Success! You completed the cooperative task!")
    else:
        print("💪 Try again to improve your cooperation!")

if __name__ == "__main__":
    play_game()