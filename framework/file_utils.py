from pathlib import Path
import os

def create_dir_if_not_exist(dir):
    a = Path(dir)
    if not a.exists():
        os.mkdir(str(a))