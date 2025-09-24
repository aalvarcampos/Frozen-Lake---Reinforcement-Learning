#############################
#    CDS403 FINAL PROJECT   # -> Part2. Non Machine Learning Method
#############################

# Andrea �lvarez Campos G01533756


import gymnasium as gym
from pynput import keyboard
from parser import prepare_for_env
from map_editor import reset_map
import csv

# Correct key-action mappings for FrozenLake
key_action_map = {
    keyboard.Key.up: 3,    # Move Up
    keyboard.Key.down: 1,  # Move Down
    keyboard.Key.left: 0,  # Move Left
    keyboard.Key.right: 2  # Move Right
}

current_action = None
exit_game = False
episode_data = []  # To store episode-level data

# Boolean to control render mode
fast_mode = True  # Set to False for visualizing the game

def on_press(key):
    global current_action, exit_game
    if key in key_action_map:
        current_action = key_action_map[key]
    elif key == keyboard.Key.esc:
        print("Exiting the game...")
        exit_game = True

def save_to_csv(data, filename="frozen_lake_NonMachineLearningMethod.csv"):
    """Save the consolidated episode data to a CSV file."""
    fieldnames = ["episode", "total_reward", "steps", "terminated"]
    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def calculate_reward(current_distance, next_distance, next_pos, goal_pos, obstacle_positions, visited_positions, position_history):
    if next_pos == goal_pos:
        return 1000
    if next_pos in obstacle_positions:
        return -100
    if next_pos in visited_positions:
        return -30
    if next_distance < current_distance:
        return 50
    elif next_distance > current_distance:
        return -50
    return -1

def get_automatic_action(current_pos, target_pos, map_size, lake_map, visited_positions, tried_directions, backtrack_stack):
    current_row, current_col = divmod(current_pos, map_size)
    target_row, target_col = divmod(target_pos, map_size)

    # Prioritize movement towards the target
    prioritized_directions = [
        (3, current_row > target_row, (current_row - 1, current_col)),  # Move Up
        (1, current_row < target_row, (current_row + 1, current_col)),  # Move Down
        (2, current_col < target_col, (current_row, current_col + 1)),  # Move Right
        (0, current_col > target_col, (current_row, current_col - 1))   # Move Left
    ]

    for action, condition, (next_row, next_col) in prioritized_directions:
        next_pos = next_row * map_size + next_col
        if condition and 0 <= next_row < map_size and 0 <= next_col < map_size:
            if lake_map[next_row][next_col] == 'G':
                return action
            if lake_map[next_row][next_col] != 'H' and next_pos not in visited_positions:
                visited_positions.add(next_pos)
                backtrack_stack.append(current_pos)
                return action

    for action, _, (next_row, next_col) in prioritized_directions:
        next_pos = next_row * map_size + next_col
        if 0 <= next_row < map_size and 0 <= next_col < map_size:
            if lake_map[next_row][next_col] != 'H' and next_pos not in visited_positions:
                visited_positions.add(next_pos)
                backtrack_stack.append(current_pos)
                return action

    if backtrack_stack:
        previous_pos = backtrack_stack.pop()
        previous_row, previous_col = divmod(previous_pos, map_size)
        if previous_row < current_row:
            return 3  # Move Up
        elif previous_row > current_row:
            return 1  # Move Down
        elif previous_col < current_col:
            return 0  # Move Left
        elif previous_col > current_col:
            return 2  # Move Right

    return 0

# Prepare the map
map_file_path = "map_1.txt"
prepared_map = prepare_for_env(map_file_path)

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", desc=prepared_map, is_slippery=False, render_mode=None if fast_mode else "human")

try:
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    num_episodes = 1000
    max_steps_per_episode = 200
    chest_position = 15
    map_size = len(prepared_map)

    for episode in range(num_episodes):
        if exit_game:
            env.close()
            listener.stop()
            break

        observation, info = env.reset(seed=42)
        print(f"Starting Episode {episode + 1} | Starting Observation: {observation}")

        visited_positions = set()
        tried_directions = set()
        backtrack_stack = []
        position_history = []
        total_reward = 0
        steps = 0
        terminated = False

        for step in range(max_steps_per_episode):
            if exit_game:
                env.close()
                listener.stop()
                break

            if not fast_mode:
                env.render()

            current_distance = abs(chest_position - observation)
            action = get_automatic_action(
                observation, chest_position, map_size,
                prepared_map, visited_positions, tried_directions, backtrack_stack
            )

            next_observation, _, terminated, truncated, info = env.step(action)
            next_distance = abs(chest_position - next_observation)

            reward = calculate_reward(
                current_distance, next_distance,
                next_observation, chest_position,
                set(), visited_positions, position_history
            )
            total_reward += reward
            steps += 1

            observation = next_observation

            if terminated or truncated:
                print("Episode finished")
                reset_map(map_file_path)
                prepared_map = prepare_for_env(map_file_path)
                chest_position = 15
                break

        # Store episode-level data
        episode_data.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "terminated": terminated
        })

    save_to_csv(episode_data, filename="frozen_lake_NonMachineLearningMethod.csv")
    print("Episode summary saved to frozen_lake_NonMachineLearningMethod.csv")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    listener.stop()
