import timm
import torch
import numpy as np
from plantclef.model_setup import setup_fine_tuned_model
from plantclef.serde import deserialize_image

from .params import HasModelName, HasModelPath, HasBatchSize

from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType


class EmbedderFineTunedDINOv2(
    Transformer,
    HasInputCols,
    HasOutputCols,
    HasModelPath,
    HasModelName,
    HasBatchSize,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Wrapper for the fine-tuned DINOv2 model for extracting embeddings.
    """

    def __init__(
        self,
        input_cols: list = ["leaf_mask", "flower_mask", "plant_mask"],
        output_cols: list = ["leaf_embed", "flower_embed", "plant_embed"],
        model_path: str = setup_fine_tuned_model(),
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size: int = 8,
        grid_size: int = 4,
    ):
        super().__init__()
        self._setDefault(
            inputCols=input_cols,
            outputCols=output_cols,
            modelPath=model_path,
            modelName=model_name,
            batchSize=batch_size,
        )
        self.num_classes = 7806  # total number of plant species
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(
            self.getModelName(),
            pretrained=False,
            num_classes=self.num_classes,
            checkpoint_path=self.getModelPath(),
        )
        # Data transform
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=False
        )
        # Move model to GPU if available
        self.model.to(self.device)
        self.model.eval()
        self.grid_size = grid_size

    def _split_into_grid(self, mask_array):
        """Splits the numpy mask array into grid tiles."""
        h, w = mask_array.shape
        grid_h, grid_w = h // self.grid_size, w // self.grid_size
        tiles = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                tile = mask_array[
                    i * grid_h : (i + 1) * grid_h, j * grid_w : (j + 1) * grid_w
                ]
                tiles.append(tile)
        return tiles

    def _nvidia_smi(self):
        from subprocess import run, PIPE

        try:
            result = run(
                ["nvidia-smi"], check=True, stdout=PIPE, stderr=PIPE, text=True
            )
            print("=== GPU Utilization (before/after prediction) ===")
            print(result.stdout)
        except Exception as e:
            print(f"nvidia-smi failed: {e}")

    def _make_predict_fn(self):
        """Return UDF using a closure over the model"""

        # check on the nvidia stats when generating the predict function
        self._nvidia_smi()

        def predict(input_data):
            # TODO: remove this print statement, debugging only
            print(f"[DEBUG] input type: {type(input_data)}, length: {len(input_data)}")

            img = deserialize_image(input_data).convert("RGB")
            tiles = self._split_into_grid(np.array(img))
            results = []
            for tile in tiles:
                processed_tile = self.transforms(tile).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.model.forward_features(processed_tile)
                    cls_token = features[:, 0, :].squeeze(0)
                cls_embeddings = cls_token.cpu().numpy().tolist()
                results.append(cls_embeddings)
            return results

        return predict

    def _transform(self, df: DataFrame):
        predict_fn = self._make_predict_fn()
        predict_udf = F.udf(predict_fn, ArrayType(ArrayType(FloatType())))
        # retrieve embeddings for each input column
        for idx, (input_col, output_col) in enumerate(
            zip(self.getInputCols(), self.getOutputCols())
        ):
            intermediate_col = f"all_{output_col}"
            df = df.withColumn(intermediate_col, predict_udf(F.col(input_col))).drop(
                input_col
            )
            # explode embeddings so that each row has a single tile embedding
            if idx == 0:  # only explode the tile column once
                df = df.selectExpr(
                    "*", f"posexplode({intermediate_col}) as (tile, {output_col})"
                )
            else:
                df = df.selectExpr(
                    "*", f"posexplode({intermediate_col}) as (tmp_tile, {output_col})"
                )
                df = df.drop(intermediate_col).drop("tmp_tile")

        return df
