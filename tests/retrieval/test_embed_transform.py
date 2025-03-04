import pytest

from plantclef.retrieval.embed.transform import EmbedderFineTunedDINOv2
from plantclef.model_setup import setup_fine_tuned_model


@pytest.mark.parametrize(
    "model_name, expected_dim, grid_size",
    [
        ("vit_base_patch14_reg4_dinov2.lvd142m", 768, 4),
    ],
)
def test_embedder_finetuned_dinov2(
    spark,
    model_name,
    expected_dim,
    grid_size,
    temp_joined_parquet,
):
    # join the test data with the mask data
    df = spark.read.parquet(temp_joined_parquet.as_posix())
    df.printSchema()
    print(f"df count: {df.count()}", flush=True)  # 1 row
    # run model
    model = EmbedderFineTunedDINOv2(
        input_cols=["data"],
        output_cols=["cls_embedding"],
        model_path=setup_fine_tuned_model(),
        model_name=model_name,
        batch_size=1,
        grid_size=grid_size,
    )
    transformed = model.transform(df).cache()
    transformed.printSchema()

    # one image, 4x4 grid size
    count = transformed.count()
    print(f"count: {count}", flush=True)
    assert transformed.count() == grid_size * grid_size
    assert transformed.columns == ["image_name", "tile", "cls_embedding"]

    tile_values = set([row.tile for row in transformed.select("tile").collect()])
    assert tile_values == set(range(grid_size * grid_size))

    row = transformed.select("cls_embedding").first()
    assert isinstance(row.cls_embedding, list)
    assert len(row.cls_embedding) == expected_dim
    assert all(isinstance(x, float) for x in row.cls_embedding)
