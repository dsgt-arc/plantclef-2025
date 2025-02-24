import os
from pathlib import Path


def get_model_dir() -> str:
    """
    Get the model directory in the plantclef shared project for the current user on PACE
    """
    home_dir = Path(os.path.expanduser("~"))
    return f"{home_dir}/p-dsgt_clef2025-0/shared/plantclef/models"


def get_scratch_model_dir() -> str:
    """
    Get the model directory in the plantclef shared project for the current user on PACE
    """
    home_dir = Path(os.path.expanduser("~"))
    return f"{home_dir}/scratch/plantclef/models"


def setup_fine_tuned_model(
    scratch_model: bool = True,
    use_only_classifier: bool = False,
):
    """
    Downloads and unzips a model from PACE and returns the path to the specified model file.
    Checks if the model already exists and skips download and extraction if it does.

    :return: Absolute path to the model file.
    """
    # get directory for model
    if scratch_model:
        model_base_path = get_scratch_model_dir()
    else:
        model_base_path = get_model_dir()
    tar_filename = "model_best.pth.tar"
    pretrained_model = (
        "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all"
    )
    if use_only_classifier:
        pretrained_model = "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier"
    relative_model_path = f"pretrained_models/{pretrained_model}/{tar_filename}"
    full_model_path = os.path.join(model_base_path, relative_model_path)

    # Check if the model file exists
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model file not found at: {full_model_path}")

    # Return the path to the model file
    return full_model_path


def setup_segment_anything_checkpoint_path():
    home_dir = Path(os.path.expanduser("~"))
    sam_checkpoint_path = os.path.join(
        home_dir, "scratch/plantclef/SAM_checkpoint/weights", "sam_vit_h_4b8939.pth"
    )
    return sam_checkpoint_path


def setup_groundingdino_checkpoint_path():
    home_dir = Path(os.path.expanduser("~"))
    grounding_dino_checkpoint_path = os.path.join(
        home_dir,
        "scratch/plantclef/groundingdino/checkpoint/weights",
        "groundingdino_swint_ogc.pth",
    )
    return grounding_dino_checkpoint_path


def setup_groundingdino_config_path():
    home_dir = Path(os.path.expanduser("~"))
    grounding_dino_config_path = os.path.join(
        home_dir,
        "scratch/plantclef/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    return grounding_dino_config_path


if __name__ == "__main__":
    # Get model
    dino_model_path = setup_fine_tuned_model()
    print("Model path:", dino_model_path)

    # Get SAM checkpoint path
    sam_checkpoint_path = setup_segment_anything_checkpoint_path()
    print("SAM checkpoint path:", sam_checkpoint_path)

    # Get GroundingDINO checkpoint path
    grounding_dino_checkpoint_path = setup_groundingdino_checkpoint_path()
    print("GroundingDINO checkpoint path:", grounding_dino_checkpoint_path)

    # Get GroundingDINO config path
    grounding_dino_config_path = setup_groundingdino_config_path()
    print("GroundingDINO config path:", grounding_dino_config_path)
