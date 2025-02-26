import io
import torch
import numpy as np
from PIL import Image
from plantclef.masking.transform import WrappedMasking


def test_wrapped_mask_detect(test_image):
    model = WrappedMasking()
    detections = model.detect(test_image)

    assert "boxes" in detections
    assert "scores" in detections
    assert "text_labels" in detections
    # TODO: check that there are detections on the test image
    assert len(detections["boxes"]) > 0
    assert len(detections["scores"]) > 0
    assert len(detections["text_labels"]) > 0


def test_wrapped_mask_segment(test_image):
    # TODO: implement
    model = WrappedMasking()
    detections = model.detect(test_image)
    input_boxes = torch.tensor(
        detections["boxes"].cpu().numpy(), dtype=torch.float32
    ).unsqueeze(0)
    masks = model.segment(test_image, input_boxes=input_boxes)
    assert len(masks) > 0


def test_empty_png(test_image):
    expected_shape = test_image.size[::-1]  # extract (H, W) from (W, H)
    model = WrappedMasking()
    empty_image = model.empty_png(test_image.size[::-1])
    assert isinstance(empty_image, (bytes, bytearray))
    img = Image.open(io.BytesIO(empty_image))
    assert img.mode == "RGB"
    img_array = np.array(img)
    assert np.all(img_array == 0)
    assert img_array.shape == (expected_shape[0], expected_shape[1], 3)


def test_group_masks_by_class():
    model = WrappedMasking()

    # simulated masks (3 masks of shape (4,4))
    masks = np.array(
        [
            [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1], [0, 1, 0, 0]],  # Mask 1
            [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 1]],  # Mask 2
            [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 0], [0, 0, 0, 1]],  # Mask 3
        ]
    )

    # simulated scores for 3 classes
    scores = torch.tensor(
        [
            [0.2, 0.5, 0.3],  # Mask 1 → Highest score at class 1
            [0.1, 0.7, 0.2],  # Mask 2 → Highest score at class 1
            [0.3, 0.2, 0.5],  # Mask 3 → Highest score at class 2
        ]
    )  # Shape: (3, 3) → Each mask gets a class assignment

    grouped_masks = model.group_masks_by_class(masks, scores)

    # ensure the masks are grouped correctly
    assert isinstance(grouped_masks, dict), "Output should be a dictionary"
    assert len(grouped_masks) > 0, "Grouped masks should not be empty"
    assert 1 in grouped_masks, "Class 1 should be present in grouped masks"
    assert 2 in grouped_masks, "Class 2 should be present in grouped masks"
    assert len(grouped_masks[1]) == 2, "Two masks should be assigned to Class 1"
    assert len(grouped_masks[2]) == 1, "One mask should be assigned to Class 2"


def test_merge_masks():
    model = WrappedMasking()

    # simulated empty image
    empty_image = model.empty_png((4, 4))  # 4x4 empty image

    # simulated grouped masks for 2 classes
    grouped_masks = {
        0: [
            np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1], [0, 1, 0, 0]]),
            np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 1]]),
        ],
        1: [
            np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 0], [0, 0, 0, 1]]),
        ],
    }

    # merge masks
    final_mask_bytes, class_mask_results = model.merge_masks(grouped_masks, empty_image)

    # ensure the function returns bytes
    assert isinstance(
        final_mask_bytes, (bytes, bytearray)
    ), "Final mask should be bytes"

    # ensure all expected class masks are present
    assert "leaf" in class_mask_results
    assert "flower" in class_mask_results
    assert "plant" in class_mask_results

    # ensure each mask is in bytes format
    for mask in class_mask_results.values():
        assert isinstance(mask, (bytes, bytearray)), "Each class mask should be bytes"


def test_wrapped_mask(spark_df):
    model = WrappedMasking(input_col="img", output_col="masks", batch_size=1)
    transformed = model.transform(spark_df).cache()
    transformed.printSchema()
    transformed.show()
    transformed.count()

    assert transformed.count() == 1
    assert "masks" in transformed.columns

    row = transformed.select("masks").first()
    assert isinstance(row.masks["combined_mask"], (bytes, bytearray))
