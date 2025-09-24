#############################
#    CDS403 FINAL PROJECT   # -> Part1. Initialization
#############################

# Andrea Alvarez Campos G01533756

import gymnasium as gym
from pynput import keyboard
from parser import prepare_for_env
# Add import for the reset function
from map_editor import reset_map
import csv

# Correct key-action mappings for FrozenLake
key_action_map = {
    keyboard.Key.up: 3,    # Move Up
    keyboard.Key.down: 1,  # Move Down
    keyboard.Key.left: 0,  # Move Left
    keyboard.Key.right: 2  # Move Right
}

current_action = None  # To store the last action selected by the user
exit_game = False  # To track if the game should exit
experiences = []  # To store state-action pairs

def on_press(key):
    global current_action, exit_game
    if key in key_action_map:
        current_action = key_action_map[key]  # Update the action based on key press
    elif key == keyboard.Key.esc:  # Check if the Esc key is pressed
        print("Exiting the game...")
        exit_game = True

def save_to_csv(data, filename="frozen_lake_experiences.csv"):
    """Save the collected data to a CSV file."""
    fieldnames = ["episode", "step", "state", "action", "reward", "next_state"]
    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Prepare the map using the provided file
map_file_path = "map_1.txt"  # Ensure it's in the same directory
prepared_map = prepare_for_env(map_file_path)

# Create the FrozenLake environment with the custom map and slippery flag
env = gym.make("FrozenLake-v1", desc=prepared_map, is_slippery=False, render_mode="human")

# Start a keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Main game loop modification
try:
    num_episodes = 10  # Number of episodes to play
    max_steps_per_episode = 50  # Max steps per episode
    total_experiences = 0

    for episode in range(num_episodes):
        if exit_game:
            break  # Exit the game if Esc is pressed

        observation, info = env.reset(seed=42)
        print(f"Starting Episode {episode + 1} | Starting Observation: {observation}")

        for step in range(max_steps_per_episode):
            if exit_game:
                break  # Exit the game if Esc is pressed

            env.render()  # Render the environment

            if current_action is not None:  # Execute the selected action
                action = current_action
                current_action = None  # Reset the action
            else:
                print("Press an arrow key to move or 'Esc' to exit!")
                continue

            # Perform the action and get the results
            next_observation, reward, terminated, truncated, info = env.step(action)

            # Check if the goal (treasure chest) is reached
            if reward > 0:  # Assuming reaching the chest gives a positive reward
                print("Congratulations! The treasure chest has been reached.")
                
                # Reset the map and reload the environment
                reset_map(map_file_path)  # Reset the map
                prepared_map = prepare_for_env(map_file_path)  # Reload the map
                env = gym.make("FrozenLake-v1", desc=prepared_map, is_slippery=False, render_mode="human")  # Recreate the environment
                
                break  # End the current episode


            # Store the experience
            experiences.append({
                "episode": episode,
                "step": step,
                "state": observation,
                "action": action,
                "reward": reward,
                "next_state": next_observation
            })
            total_experiences += 1

            # Update the current observation
            observation = next_observation

            if terminated or truncated:
                print("Episode finished")
                break

    # Save all experiences to a CSV file
    save_to_csv(experiences)
    print(f"Saved {len(experiences)} experiences to frozen_lake_experiences.csv")

    env.close()

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    listener.stop()  # Ensure the keyboard listener is stopped

