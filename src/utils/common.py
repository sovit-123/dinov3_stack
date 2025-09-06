import os
import sys

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def get_dinov3_paths():
    """
    Get DINOv3 repository and weights paths from environment variables.
    
    Returns:
        tuple: (repo_path, weights_path) as Path objects
        
    Raises:
        FileNotFoundError: If paths are not set in .env or don't exist
        ValueError: If environment variables are not set
    """
    # Get paths from environment
    repo_path_str = os.getenv('DINOv3_REPO')
    weights_path_str = os.getenv('DINOv3_WEIGHTS')
    
    # Check if environment variables are set
    if not repo_path_str:
        raise ValueError(
            'DINOV3_REPO not found in environment variables. '
            'Please set DINOV3_REPO=/path/to/dinov3/repo in your .env file'
        )
    
    if not weights_path_str:
        raise ValueError(
            'DINOV3_WEIGHTS not found in environment variables. '
            'Please set DINOV3_WEIGHTS=/path/to/weights in your .env file'
        )
    
    # Convert to Path objects
    repo_path = Path(repo_path_str).expanduser().resolve()
    weights_path = Path(weights_path_str).expanduser().resolve()
    
    # Verify paths exist
    if not repo_path.exists():
        raise FileNotFoundError(
            f"DINOv3 repository not found at: {repo_path}\n"
            f"Please check your DINOV3_REPO path in .env file"
        )
    
    if not weights_path.exists():
        raise FileNotFoundError(
            f"DINOv3 weights not found at: {weights_path}\n"
            f"Please check your DINOV3_WEIGHTS path in .env file"
        )
    
    return repo_path_str, weights_path_str


if __name__ == '__main__':
    try:
        repo, weights = get_dinov3_paths()
        print(f'DINOv3 Repository: {repo}')
        print(f'DINOv3 Weights: {weights}')
    except (ValueError, FileNotFoundError) as e:
        print(f'Error: {e}')