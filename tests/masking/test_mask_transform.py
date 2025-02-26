import torch
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
    # raise NotImplementedError("check this returns masks")
    model = WrappedMasking()
    detections = model.detect(test_image)
    input_boxes = torch.tensor(
        detections["boxes"].cpu().numpy(), dtype=torch.float32
    ).unsqueeze(0)
    masks = model.segment(test_image, input_boxes=input_boxes)
    assert len(masks) > 0


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
