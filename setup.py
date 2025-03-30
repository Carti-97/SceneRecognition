import os
import shutil
from config import DATA_DIR, CHECKPOINT_DIR, ROOT_DIR

def setup_directory_structure():
    """
    Initialize project directory structure
    """
    print("Initializing project directory structure...")
    
    # Create data directories
    os.makedirs(os.path.join(DATA_DIR, 'SUN397_split', 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'SUN397_split', 'val'), exist_ok=True)
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"The following directories have been created:")
    print(f"- Data directory: {DATA_DIR}")
    print(f"- Checkpoint directory: {CHECKPOINT_DIR}")
    
    print("\nProject structure initialization completed.")
    print("Next steps:")
    print("1. Copy dataset to data/SUN397_split directory")
    print("2. Run 'python train.py' to start training")
    print("3. Run 'python eval.py' to evaluate model")

if __name__ == "__main__":
    setup_directory_structure() 