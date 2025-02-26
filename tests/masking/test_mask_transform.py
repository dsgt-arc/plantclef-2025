from plantclef.masking.transform import WrappedMasking


def test_wrapped_mask_detect(test_image):
    model = WrappedMasking()
    detections = model.detect(test_image)
    assert "boxes" in detections
    assert "scores" in detections
    assert "text_labels" in detections
    # TODO: check that there are detections on the test image


def test_wrapped_mask_segment(test_image):
    # TODO: implement
    raise NotImplementedError("check this returns masks")


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
