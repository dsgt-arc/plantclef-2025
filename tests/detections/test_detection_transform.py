from plantclef.detection.transform import WrappedGroundingDINO


def test_wrapped_groundingdino(spark_df):
    model = WrappedGroundingDINO(
        input_col="data",
        output_col="output",
        batch_size=2,
        checkpoint_path_groundingdino="IDEA-Research/grounding-dino-base",
    )
    transformed = model.transform(spark_df).cache()
    transformed.printSchema()
    transformed = transformed.select("image_name", "output.*")

    assert transformed.count() == 1  # one image
    assert transformed.columns == [
        "image_name",
        "extracted_bbox",
        "boxes",
        "scores",
        "text_labels",
    ]

    row = transformed.first()
    assert isinstance(row.extracted_bbox, list)
    assert all(isinstance(x, bytearray) for x in row.extracted_bbox)
    assert isinstance(row.boxes, list)
    assert all(
        isinstance(box, list) and all(isinstance(coord, int) for coord in box)
        for box in row.boxes
    )
    assert isinstance(row.scores, list)
    assert all(isinstance(x, float) for x in row.scores)
    assert isinstance(row.text_labels, list)
    assert all(isinstance(x, str) for x in row.text_labels)


# Run this script
# pytest -vv -s test_detection_transform.py
