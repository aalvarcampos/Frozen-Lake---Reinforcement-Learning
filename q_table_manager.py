#############################
#    CDS403 FINAL PROJECT   # -> Q-table manager
#############################

# Andrea Alvarez Campos G01533756

import os
import numpy as np

class QTableManager:
    def __init__(self, actions, file_path="q_table.txt"):
        """
        Initialize a QTableManager with actions and file I/O for Q-table in text format.
        :param actions: List of possible actions.
        :param file_path: Path to save/load the Q-table as a .txt file.
        """
        self.actions = actions
        self.file_path = file_path
        self.q_table = self._load_or_initialize()

    def _load_or_initialize(self):
        """
        Load the Q-table from a text file or initialize it with zeros.
        """
        if os.path.exists(self.file_path):
            print(f"[INFO]: Loading Q-table from {self.file_path}")
            return np.loadtxt(self.file_path)
        else:
            print("[INFO]: Initializing new Q-table")
            num_states = 16  # Example for a 4x4 FrozenLake grid
            num_actions = len(self.actions)
            return np.zeros((num_states, num_actions))

    def save_q_table(self):
        """
        Save the Q-table to a text file in scientific notation format.
        """
        np.savetxt(self.file_path, self.q_table, fmt="%.18e")
        print(f"[INFO]: Q-table saved to {self.file_path}")

    def load_q_table(self):
        """
        Load the Q-table from a text file. If the file does not exist, raise an error.
        """
        if os.path.exists(self.file_path):
            print(f"[INFO]: Reloading Q-table from {self.file_path}")
            self.q_table = np.loadtxt(self.file_path)
        else:
            raise FileNotFoundError(f"Q-table file {self.file_path} not found!")

    def get_q_value(self, state, action):
        """
        Get the Q-value for a given state and action.
        :param state: State index.
        :param action: Action index.
        :return: Q-value for the state-action pair.
        """
        return self.q_table[state, action]

    def update_q_value(self, state, action, value):
        """
        Update the Q-value for a given state and action.
        :param state: State index.
        :param action: Action index.
        :param value: New Q-value to be updated.
        """
        self.q_table[state, action] = value

    def get_best_action(self, state):
        """
        Get the best action for a given state based on the Q-table.
        :param state: State index.
        :return: Action index with the highest Q-value.
        """
        return np.argmax(self.q_table[state, :])
