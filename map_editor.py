#############################
#    CDS403 FINAL PROJECT   # -> Map editor
#############################

# Andrea Alvarez Campos G01533756

import random

# Function to reset the map
def reset_map(file_path):
    try:
        # Read the content of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Convert the lines into a list of lists for easier modification
        map_grid = [list(line.strip()) for line in lines]

        # Find all positions of 'F' for random placement
        f_positions = [(i, j) for i, row in enumerate(map_grid) for j, char in enumerate(row) if char == 'F']

        if not f_positions:
            raise ValueError("No valid positions ('F') to reset the game.")

        # Randomly select a new position for 'G' and 'S'
        g_position = random.choice(f_positions)
        s_position = random.choice(f_positions)

        # Ensure 'G' and 'S' are not the same
        while g_position == s_position:
            s_position = random.choice(f_positions)

        # Update the map with new positions
        for i, row in enumerate(map_grid):
            for j in range(len(row)):
                if map_grid[i][j] == 'G':
                    map_grid[i][j] = 'F'
                if map_grid[i][j] == 'S':
                    map_grid[i][j] = 'F'
        map_grid[g_position[0]][g_position[1]] = 'G'
        map_grid[s_position[0]][s_position[1]] = 'S'

        # Write the updated map back to the file
        updated_map = "\n".join("".join(row) for row in map_grid)
        with open(file_path, 'w') as file:
            file.write(updated_map)

        print("The map has been reset.")
    except Exception as e:
        print(f"An error occurred during map reset: {e}")
