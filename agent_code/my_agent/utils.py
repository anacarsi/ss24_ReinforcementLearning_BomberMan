import os
import sys

def get_settings_dir() -> str:
    """
    Get the directory of the settings file.

    :return: The directory of the settings file.
    """
    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

    sys.path.insert(0, parent_dir)

    return parent_dir
