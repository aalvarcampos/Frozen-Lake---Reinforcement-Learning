#############################
#    CDS403 FINAL PROJECT   # -> Part3. Q-learning Final Code
#############################

# Andrea Alvarez Campos G01533756

import gymnasium as gym
from parser import prepare_for_env
from map_editor import reset_map
from q_table_manager import QTableManager
import numpy as np
import time

# Configuration
is_training = False  # Set to True for training mode, False for play mode

# Hyperparameters for Q-learning
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0 if is_training else 0.01  # Start at 1.0 for training, fixed at 0.01 for playing
min_epsilon = 0.01  # Minimum value of epsilon
epsilon_decay = 0.999  # Decay rate for epsilon during training

# Load the map file
map_file_path = "map_1.txt"
prepared_map = prepare_for_env(map_file_path)

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", desc=prepared_map, is_slippery=False, render_mode="human" if not is_training else None)

# Initialize actions
actions = [0, 1, 2, 3]  # Left, Down, Right, Up

# Define state space size
grid_size = (12, 12)  # Map dimensions (12x12)
state_space_size = grid_size[0] * grid_size[1]  # Maximum possible distance states

# Initialize Q-table manager with adjusted Q-table size
q_manager = QTableManager(actions, file_path="q_table.txt")
if not is_training:
    q_manager.load_q_table()  # Load the Q-table in play mode
else:
    q_manager.q_table = np.zeros((state_space_size, len(actions)))  # Q-table for distance states x 4 actions

# Helper functions
def calculate_distance(agent_pos, goal_pos):
    """
    Calculate the Manhattan distance between the agent and the goal.
    """
    return abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])

def encode_state(agent_pos, goal_pos):
    """
    Encode the state based on the Manhattan distance between the agent and the goal.
    """
    distance = calculate_distance(agent_pos, goal_pos)
    return distance

def find_goal_position(grid, goal_char='G'):
    """
    Find the position of the goal in the grid.
    """
    for i, row in enumerate(grid):
        for j, char in enumerate(row):
            if char == goal_char:
                return (i, j)
    raise ValueError(f"Goal '{goal_char}' not found in the map!")

def find_obstacle_positions(grid, obstacle_char='H'):
    """
    Find all positions of obstacles in the grid.
    """
    obstacles = set()
    for i, row in enumerate(grid):
        for j, char in enumerate(row):
            if char == obstacle_char:
                obstacles.add((i, j))
    return obstacles

def is_valid_move(agent_pos, action, grid_size, obstacle_positions, visited_positions):
    """
    Check if a move is valid (does not lead to an obstacle, out of bounds, or a visited position).
    """
    rows, cols = grid_size
    row, col = agent_pos

    # Determine the new position based on the action
    if action == 0:  # Left
        col -= 1
    elif action == 1:  # Down
        row += 1
    elif action == 2:  # Right
        col += 1
    elif action == 3:  # Up
        row -= 1

    # Check if the new position is out of bounds, an obstacle, or already visited
    if (row < 0 or row >= rows or col < 0 or col >= cols or 
        (row, col) in obstacle_positions or (row, col) in visited_positions):
        return False
    return True

def force_move_towards_goal(agent_pos, goal_pos):
    """
    Determine the action to move directly towards the goal when distance is 1.
    """
    if agent_pos[0] < goal_pos[0]:
        return 1  # Move Down
    elif agent_pos[0] > goal_pos[0]:
        return 3  # Move Up
    elif agent_pos[1] < goal_pos[1]:
        return 2  # Move Right
    elif agent_pos[1] > goal_pos[1]:
        return 0  # Move Left

def calculate_reward(current_distance, next_distance, next_pos, goal_pos, obstacle_positions, visited_positions, position_history):
    """
    Reward the agent based on whether it moves closer or farther from the goal, penalize revisits,
    and provide a large reward for reaching the goal. Penalize loops.
    """
    # Check if the agent reached the goal
    if next_pos == goal_pos:
        return 1000  # Reward for reaching the goal

    # Check if the agent hits an obstacle
    if next_pos in obstacle_positions:
        return -100  # Strong penalty for hitting an obstacle

    # Check if the agent revisits a position
    if next_pos in visited_positions:
        return -30  # Penalty for revisiting a position

    # Check for loops in the position history
    history_length = 5  # Number of steps to consider for a loop
    if len(position_history) >= history_length * 2:
        if position_history[-history_length:] == position_history[-2 * history_length:-history_length]:
            return -50  # Penalty for being stuck in a loop

    # Reward based on proximity to the goal
    if next_distance < current_distance:
        return 50  # Higher reward for getting closer
    elif next_distance > current_distance:
        return -20  # Stronger penalty for moving farther
    else:
        return -1  # Small penalty for staying the same

def decode_position(state, grid_size):
    """
    Decode the relative state index into (row, col).
    """
    rows, cols = grid_size
    row = state // cols
    col = state % cols
    return (row, col)

# Initialize goal and obstacles
goal_pos = find_goal_position(prepared_map)
obstacle_positions = find_obstacle_positions(prepared_map)

# Main game loop
num_episodes = 2000 if is_training else 10
position_history = []  # Track position history to detect loops

for episode in range(num_episodes):
    observation, info = env.reset(seed=42)
    agent_pos = decode_position(observation, grid_size)
    state = encode_state(agent_pos, goal_pos)
    visited_positions = set()  # Track visited positions during the episode
    total_reward = 0

    print(f"Starting Episode {episode + 1} | Initial Distance: {state}")

    for step in range(100):
        if not is_training:
            env.render()
            time.sleep(0.2)

        # Determine valid actions, excluding obstacles and visited positions
        valid_actions = [action for action in actions if is_valid_move(agent_pos, action, grid_size, obstacle_positions, visited_positions)]

        # Force move towards the goal if the distance is 1
        distance_to_goal = calculate_distance(agent_pos, goal_pos)
        if distance_to_goal == 1:
            action = force_move_towards_goal(agent_pos, goal_pos)
        else:
            # Handle the case where no valid actions are available
            if not valid_actions:
                print("No valid actions available. Forcing a random action.")
                action = np.random.choice(actions)
            else:
                # Epsilon-greedy action selection with valid actions only
                if is_training:
                    if np.random.rand() < epsilon:
                        action = np.random.choice(valid_actions)  # Explore only valid actions
                    else:
                        best_action = q_manager.get_best_action(state)
                        action = best_action if best_action in valid_actions else np.random.choice(valid_actions)
                else:
                    # Play mode: Only exploit the Q-table with the updated state
                    best_action = q_manager.get_best_action(state)
                    action = best_action if best_action in valid_actions else np.random.choice(valid_actions)
                    print(f"State: {state}, Q-values: {q_manager.q_table[state]}, Selected Action: {action}")

        # Execute action
        next_observation, _, terminated, truncated, _ = env.step(action)
        next_agent_pos = decode_position(next_observation, grid_size)
        next_state = encode_state(next_agent_pos, goal_pos)

        # Update position history
        position_history.append(next_agent_pos)
        if len(position_history) > 10:  # Mantén un tamaño fijo para el historial
            position_history.pop(0)

        # Calculate distance-based reward with revisit and loop penalty
        current_distance = state
        next_distance = next_state
        reward = calculate_reward(current_distance, next_distance, next_agent_pos, goal_pos, obstacle_positions, visited_positions, position_history)
        total_reward += reward

        # Check if the agent reached the goal
        if next_agent_pos == goal_pos:
            print(f"Goal reached at step {step + 1}!")
            reset_map(map_file_path)  # Reset the map for new positions
            prepared_map = prepare_for_env(map_file_path)
            goal_pos = find_goal_position(prepared_map)
            obstacle_positions = find_obstacle_positions(prepared_map)
            env = gym.make("FrozenLake-v1", desc=prepared_map, is_slippery=False, render_mode="human" if not is_training else None)
            state = encode_state(agent_pos, goal_pos)  # Recalculate state for the new goal position
            break

        # Update visited positions
        visited_positions.add(next_agent_pos)

        # Update state
        state = next_state
        agent_pos = next_agent_pos

        if is_training:
            # Update Q-table
            old_value = q_manager.get_q_value(state, action)
            future_reward = max(q_manager.get_q_value(next_state, a) for a in actions)
            new_value = old_value + alpha * (reward + gamma * future_reward - old_value)
            q_manager.update_q_value(state, action, new_value)
            print(f"Updated Q-value for state {state}, action {action}: {new_value}")

        print(f"Step: {step + 1}, Action: {action}, Distance: {next_distance}, Reward: {reward}")

        if terminated or truncated:
            break

    # Gradually reduce epsilon during training
    if is_training:
        epsilon = max(min_epsilon, epsilon_decay * epsilon)

    print(f"Ending Episode {episode + 1} | Total Reward: {total_reward}")

# Save Q-table at the end of training
if is_training:
    q_manager.save_q_table()
else:
    print("[INFO]: Game session complete. Q-table was used for playing.")
