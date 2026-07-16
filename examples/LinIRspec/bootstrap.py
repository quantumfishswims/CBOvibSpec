# examples/scenario_1/bootstrap.py
import sys
from pathlib import Path

def patch_sys_path():
    # Start from the folder where this bootstrap.py file lives
    current_dir = Path(__file__).resolve().parent
    
    # Climb up the folder tree (up to 10 levels) to find the project root containing 'src'
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "src").is_dir():
            root_path = str(parent)
            if root_path not in sys.path:
                sys.path.insert(0, root_path)
            return
            
    raise RuntimeError("Could not find the project root containing 'src/' directory.")

patch_sys_path()