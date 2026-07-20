#!/usr/bin/env python3

import sys
from pathlib import Path

def patch_sys_path():
    current_dir = Path(__file__).resolve().parent
    
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "src").is_dir():
            root_path = str(parent)
            if root_path not in sys.path:
                sys.path.insert(0, root_path)
            return
            
    raise RuntimeError("Could not find project root containing 'src/' directory.")

patch_sys_path()