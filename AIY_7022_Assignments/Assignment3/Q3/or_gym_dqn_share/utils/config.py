import torch, os, random, numpy as np

# General Config
NUM_EPISODES = 10000
MAX_STEPS = 10
SEED = 42

# Model paths
os.makedirs("models", exist_ok=True)
MODEL_PATH_TASK1 = "models/dqn_terminal_wealth.pt"
MODEL_PATH_TASK2 = "models/dqn_step_wealth.pt"

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seeding
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_env_config(seed):
    return {
        'num_assets': 5,
        'max_steps': MAX_STEPS,
        'initial_cash': 50,
        'trans_cost': 1,
        'lot_size': 2,
        'max_units': 10,
        'seed': seed
    }
