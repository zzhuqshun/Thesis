from pathlib import Path
from utils.config import Config
from utils.utils import set_seed, setup_logging
from utils.joint import training

def run_joint():
    """Run joint training for the model.
    This function initializes the configuration, sets the random seed for reproducibility,
    sets up logging, and starts the joint training process.
    """
    # Initialize configuration
    config = Config()
    config.MODE = "joint" 
    set_seed(config.SEED)
    
    config.BASE_DIR = Path.cwd() / 'joint_training'
    config.BASE_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(config.BASE_DIR)

    config.save(config.BASE_DIR / 'config.json')
    
    # Start joint training
    training(config)
    
if __name__ == '__main__':
    run_joint()
