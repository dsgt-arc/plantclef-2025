import os
from pathlib import Path


def get_index_dir() -> str:
    """
    Get the index directory in the plantclef shared project for the current user on PACE
    """
    return str(Path.home() / "p-dsgt_clef2025-0/shared/plantclef/data/faiss/train")


def get_scratch_index_dir() -> str:
    """
    Get the index directory in the plantclef scratch project for the current user on PACE
    """
    return str(Path.home() / "scratch/plantclef/data/faiss/train")


def setup_index(
    scratch_model: bool = True,
    index_name: str = "embeddings",
) -> str:
    if scratch_model:
        index_base_path = get_scratch_index_dir()
    else:
        index_base_path = get_index_dir()
    index_filename = f"{index_name}.index"
    return f"{index_base_path}/{index_filename}"