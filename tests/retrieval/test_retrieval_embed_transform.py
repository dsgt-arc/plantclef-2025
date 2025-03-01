import pytest

from plantclef.retrieval.embed.transform import EmbedderFineTunedDINOv2
from plantclef.model_setup import setup_fine_tuned_model


@pytest.mark.parametrize(
    "model_name,expected_dim",
    [
        ("vit_base_patch14_reg4_dinov2.lvd142m", 768),
    ],
)
def test_embedder_finetuned_dinov2(spark_df, model_name, expected_dim):
    model = EmbedderFineTunedDINOv2(
        input_col="img",
        output_col="transformed",
        model_path=setup_fine_tuned_model(),
        model_name=model_name,
        batch_size=2,
        use_grid=True,
        grid_size=3,
    )
    transformed = model.transform(spark_df).cache()
    transformed.printSchema()
    transformed.show()

    assert transformed.count() == 2 * 3 * 3
    assert transformed.columns == ["tile", "transformed"]

    tile_values = set([row.tile for row in transformed.select("tile").collect()])
    assert tile_values == set(range(3 * 3))

    row = transformed.select("transformed").first()
    assert isinstance(row.transformed, list)
    assert len(row.transformed) == expected_dim
    assert all(isinstance(x, float) for x in row.transformed)
