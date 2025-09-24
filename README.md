# Frozen-Lake---Reinforcement-Learning

Reinforcement learning project using the FrozenLake environment from OpenAI Gym with custom map integration. Includes manual agent control via keyboard, experience collection, Q-learning training, and performance analysis. Developed as part of the CDS403 course in George Mason University.

## 📁 Project Structure

- **FrozenLake--Reinforcement-Learning/**
  - **frozenlake/** – Source code for environment setup and agent control  
    - `FrozenLake1_Initialization.py`  
    - `parser.py`  
    - `map_editor.py`  
    - `map_1.txt`  
  - **data/** – Collected experience data  
    - `frozen_lake_experiences.csv`  
  - **notebooks/** – Jupyter notebooks for analysis and training  
    - `FrozenLake_Exploration.ipynb`  
  - **docs/** – Project presentation and documentation  
    - `FrozenLake_Presentation.pdf`  
  - `requirements.txt` – Python dependencies  
  - `README.md` – Project overview

## 🚀 How to Run

Follow these steps to set up and run the project:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/FrozenLake--Reinforcement-Learning.git
cd FrozenLake--Reinforcement-Learning
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the initialization script to collect experiences
```bash
python frozenlake/FrozenLake1_Initialization.py
```

### 5. Explore and train your agent
```bash
jupyter notebook notebooks/FrozenLake_Exploration.ipynb
```



