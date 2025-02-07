import os
from plantclef.config import get_pace_model_dir


def setup_pretrained_model(use_only_classifier: bool = False):
    """
    Downloads and unzips a model from PACE and returns the path to the specified model file.
    Checks if the model already exists and skips download and extraction if it does.

    :return: Absolute path to the model file.
    """
    model_base_path = get_pace_model_dir()  # get directory for model
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


if __name__ == "__main__":
    # Get model
    model_path = setup_pretrained_model()
    print("Model path:", model_path)
