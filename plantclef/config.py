import os
from pathlib import Path


def get_data_dir() -> str:
    """
    Get the data directory in the plantclef shared project for the current user on PACE
    """
    home_dir = Path(os.path.expanduser("~"))
    return f"{home_dir}/p-dsgt_clef2025-0/shared/plantclef/data"


def get_scratch_data_dir() -> str:
    """
    Get the data directory in the plantclef shared project for the current user on PACE
    """
    home_dir = Path(os.path.expanduser("~"))
    return f"{home_dir}/scratch/plantclef/data"


def get_home_dir():
    """Get the home directory for the current user on PACE."""
    return Path(os.path.expanduser("~"))


def get_class_mappings_file() -> str:
    """
    Get the directory containing the class mappings for the DINOv2 model.
    """
    return f"{get_data_dir()}/train/PlantCLEF2024/class_mappings.txt"
