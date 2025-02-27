import io
import numpy as np
from PIL import Image
from plantclef.masking.transform import WrappedMasking


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
    print(f"input_boxes: {input_boxes.shape}")
    # torch.Size([1, 42, 4])
    masks = model.segment(test_image, input_boxes=input_boxes)
    # (1, 42, 3, 3024, 3024)
    assert len(masks) > 0
    assert masks.shape == (42, 3024, 3024)
    print(masks)


def test_merge_masks(test_image):
    model = WrappedMasking()
    detections = model.detect(test_image)
    input_boxes = model.convert_boxes_to_tensor(detections)
    masks = model.segment(test_image, input_boxes=input_boxes)
    # assert all the masks are the same shape
    combined_mask, class_masks = model.merge_masks(
        masks, detections["text_labels"], (test_image.height, test_image.width)
    )
    for key, mask in class_masks.items():
        assert mask.shape == combined_mask.shape
        assert mask.dtype == np.uint8
    assert combined_mask.shape == (test_image.height, test_image.width)
    assert combined_mask.dtype == np.uint8


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
    print(f"combined mask type: {type(row.masks['combined_mask'])}")
    assert isinstance(row.masks["combined_mask"], bytearray)

    # decode the bytes back into a NumPy array
    mask = np.load(io.BytesIO(row.masks["combined_mask"]))

    # ensure the mask is a NumPy array
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8

    # ensure mask has the expected dimensions (same as input image)
    img_data = spark_df.select("data").first().data
    img = Image.open(io.BytesIO(img_data))
    expected_shape = img.size[::-1]
    assert mask.shape == expected_shape

    # ensure mask contains only valid binary values (0 or 1)
    unique_values = np.unique(mask)
    assert set(unique_values).issubset({0, 1})
