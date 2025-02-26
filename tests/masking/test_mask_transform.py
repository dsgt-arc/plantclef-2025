from plantclef.masking.transform import WrappedMasking


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
