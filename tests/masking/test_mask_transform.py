import numpy as np
from plantclef.masking.transform import WrappedMasking
from plantclef.serde import deserialize_mask, deserialize_image


def test_wrapped_mask_detect(test_image):
    model = WrappedMasking()
    detections = model.detect(test_image)

    n = 42
    assert detections["boxes"].shape == (n, 4)
    assert detections["scores"].shape == (n,)
    assert len(detections["text_labels"]) == n
    print(detections["scores"])
    print(detections["text_labels"])


def test_wrapped_mask_segment(test_image):
    # TODO: implement the segment test
    model = WrappedMasking()
    detections = model.detect(test_image)
    input_boxes = model.convert_boxes_to_tensor(detections)
    print(f"input boxes shape: {input_boxes.shape}")
    # torch.Size([1, 42, 4])
    masks = model.segment(test_image, input_boxes=input_boxes)
    # (1, 42, 3, 3024, 3024)
    assert len(masks) > 0
    assert masks.shape == (42, 3024, 3024)
    print(f"masks: {masks}")


def test_merge_class_masks(test_image):
    model = WrappedMasking()
    detections = model.detect(test_image)
    input_boxes = model.convert_boxes_to_tensor(detections)
    masks = model.segment(test_image, input_boxes=input_boxes)
    # assert all the masks are the same shape
    class_masks = model.merge_class_masks(
        masks, detections["text_labels"], (test_image.height, test_image.width)
    )
    print(f"all mask classes: {class_masks.keys()}")
    print(f"rock mask: {class_masks['rock']}")
    for key, mask in class_masks.items():
        assert mask.shape == class_masks["leaf"].shape
        assert mask.dtype == np.uint8
    assert class_masks["leaf"].shape == (test_image.height, test_image.width)
    assert class_masks["leaf"].dtype == np.uint8


def test_wrapped_mask(spark_df):
    model = WrappedMasking(input_col="data", output_col="masks", batch_size=1)
    transformed = model.transform(spark_df).cache()
    transformed.printSchema()
    transformed.show(vertical=True, truncate=100, n=2)
    transformed.count()

    assert transformed.count() == 1
    assert "masks" in transformed.columns

    row = transformed.select("masks").first()

    # ensure that the output mask is a NumPy array
    print(f"combined mask type: {type(row.masks['leaf_mask'])}")
    assert isinstance(row.masks["leaf_mask"], bytearray)

    # decode the bytes back into a NumPy array
    mask = deserialize_mask(row.masks["leaf_mask"])
    rock_mask = deserialize_mask(row.masks["rock_mask"])

    # ensure the mask is a NumPy array
    assert isinstance(mask, np.ndarray)
    assert isinstance(rock_mask, np.ndarray)
    assert mask.dtype == np.uint8
    assert rock_mask.dtype == np.uint8

    # ensure mask has the expected dimensions (same as input image)
    img_data = spark_df.select("data").first().data
    img = deserialize_image(img_data)
    expected_shape = img.size[::-1]
    assert mask.shape == expected_shape

    # ensure mask contains only valid binary values (0 or 1)
    unique_values = np.unique(mask)
    assert set(unique_values).issubset({0, 1})
