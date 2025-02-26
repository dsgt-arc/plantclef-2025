import io
import numpy as np
from plantclef.masking.transform import WrappedMasking


# def test_wrapped_mask_detect(test_image):
#     model = WrappedMasking()
#     detections = model.detect(test_image)

#     assert "boxes" in detections
#     assert "scores" in detections
#     assert "text_labels" in detections
#     # TODO: check that there are detections on the test image
#     assert len(detections["boxes"]) > 0
#     assert len(detections["scores"]) > 0
#     assert len(detections["text_labels"]) > 0


# def test_wrapped_mask_segment(test_image):
#     # TODO: implement the segment test
#     model = WrappedMasking()
#     detections = model.detect(test_image)
#     input_boxes = torch.tensor(
#         detections["boxes"].cpu().numpy(), dtype=torch.float32
#     ).unsqueeze(0)
#     masks = model.segment(test_image, input_boxes=input_boxes)
#     assert len(masks) > 0


# def test_empty_array(test_image):
#     expected_shape = test_image.size[::-1]  # extract (H, W)
#     model = WrappedMasking()
#     empty_bytes = model.empty_array(expected_shape)  # get raw NumPy array

#     # ensure the output is a NumPy array, not bytes
#     assert isinstance(empty_bytes, bytes)
#     # Decode the bytes back into a NumPy array
#     empty_mask = np.load(io.BytesIO(empty_bytes))  # Load NumPy array

#     # ensure the decoded mask is a NumPy array
#     assert isinstance(empty_mask, np.ndarray)
#     assert empty_mask.dtype == np.uint8
#     assert empty_mask.shape == expected_shape
#     assert np.all(empty_mask == 0)


# def test_group_masks_by_class():
#     model = WrappedMasking()

#     # simulated masks (3 masks of shape (4,4))
#     masks = np.array([
#         [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1], [0, 1, 0, 0]],  # Mask 1
#         [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 1]],  # Mask 2
#         [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 0], [0, 0, 0, 1]],  # Mask 3
#     ], dtype=np.uint8)  # ensure dtype is uint8

#     # simulated scores for 3 classes
#     scores = torch.tensor([
#         [0.2, 0.5, 0.3],  # Mask 1 → Highest score at class 1
#         [0.1, 0.7, 0.2],  # Mask 2 → Highest score at class 1
#         [0.3, 0.2, 0.5],  # Mask 3 → Highest score at class 2
#     ])  # Shape: (3, 3) → each mask gets a class assignment

#     grouped_masks = model.group_masks_by_class(masks, scores)

#     # ensure the masks are grouped correctly
#     assert isinstance(grouped_masks, dict)
#     assert len(grouped_masks) > 0
#     assert 1 in grouped_masks
#     assert 2 in grouped_masks
#     assert len(grouped_masks[1]) == 2
#     assert len(grouped_masks[2]) == 1


# def test_merge_masks():
#     model = WrappedMasking()

#     # simulated empty image
#     empty_shape = (4, 4)  # 4x4 empty mask
#     empty_mask = np.zeros(empty_shape, dtype=np.uint8)

#     # simulated grouped masks for 2 classes
#     grouped_masks = {
#         0: [
#             np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1], [0, 1, 0, 0]], dtype=np.uint8),
#             np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 1]], dtype=np.uint8),
#         ],
#         1: [
#             np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 0], [0, 0, 0, 1]], dtype=np.uint8),
#         ],
#     }

#     # merge masks
#     final_mask, class_mask_results = model.merge_masks(grouped_masks, empty_shape)

#     # ensure the function returns a NumPy array
#     assert isinstance(final_mask, np.ndarray)
#     assert final_mask.dtype == np.uint8
#     assert final_mask.shape == (4, 4)

#     # ensure all expected class masks are present
#     assert "leaf" in class_mask_results
#     assert "flower" in class_mask_results
#     assert "plant" in class_mask_results

#     # ensure each mask is a NumPy array of the same shape
#     for mask in class_mask_results.values():
#         assert isinstance(mask, np.ndarray)
#         assert mask.dtype == np.uint8
#         assert mask.shape == (4, 4)


def test_wrapped_mask(spark_df):
    model = WrappedMasking(input_col="data", output_col="masks", batch_size=1)
    transformed = model.transform(spark_df).cache()
    transformed.printSchema()
    transformed.show()
    transformed.count()

    assert transformed.count() == 1
    assert "masks" in transformed.columns

    row = transformed.select("masks").first()

    # ensure that the output mask is a NumPy array
    assert isinstance(row.masks["combined_mask"], bytes)
    # Decode the bytes back into a NumPy array
    mask = np.load(io.BytesIO(row.masks["combined_mask"]))

    # Ensure the mask is a NumPy array
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8

    # Ensure mask has the expected dimensions (same as input image)
    expected_shape = spark_df.select("img").first().img.shape[:2]
    assert mask.shape == expected_shape

    # Ensure mask contains only valid binary values (0 or 1)
    unique_values = np.unique(mask)
    assert set(unique_values).issubset({0, 1})
