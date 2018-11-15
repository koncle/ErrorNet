from pathlib import Path
import os

import shutil

def create_dir_if_not_exist(dir):
    a = Path(dir)
    if not a.exists():
        os.mkdir(str(a))


def copy_file(src, dest):
    shutil.copy(src, dest)