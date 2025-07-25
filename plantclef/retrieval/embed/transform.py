import io

import timm
import torch
from PIL import Image
from plantclef.model_setup import setup_fine_tuned_model
from plantclef.embedding.transform import HasModelPath, HasModelName
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType


class EmbedderFineTunedDINOv2(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasModelPath,
    HasModelName,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Wrapper for the fine-tuned DINOv2 model for extracting embeddings.
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        model_path: str = setup_fine_tuned_model(),
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        use_grid: bool = False,
        grid_size: int = 3,
    ):
        super().__init__()
        self._setDefault(
            inputCol=input_col,
            outputCol=output_col,
            modelPath=model_path,
            modelName=model_name,
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
        self.use_grid = use_grid
        self.grid_size = grid_size

    def _split_into_grid(self, image):
        w, h = image.size
        grid_w, grid_h = w // self.grid_size, h // self.grid_size
        images = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                left = i * grid_w
                upper = j * grid_h
                right = left + grid_w
                lower = upper + grid_h
                crop_image = image.crop((left, upper, right, lower))
                images.append(crop_image)
        return images

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
            img = Image.open(io.BytesIO(input_data))
            images = [img]
            if self.use_grid:
                images = self._split_into_grid(img)
            results = []
            for tile in images:
                processed_image = self.transforms(tile).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.model.forward_features(processed_image)
                    cls_token = features[:, 0, :].squeeze(0)
                cls_embeddings = cls_token.cpu().numpy().tolist()
                results.append(cls_embeddings)
            return results

        return predict

    def _transform(self, df: DataFrame):
        predict_fn = self._make_predict_fn()
        predict_udf = F.udf(predict_fn, ArrayType(ArrayType(FloatType())))
        intermediate_col = "all_" + self.getOutputCol()
        df = df.withColumn(
            intermediate_col, predict_udf(F.col(self.getInputCol()))
        ).drop(self.getInputCol())

        # explode embeddings so that each row has a single tile embedding
        df = df.selectExpr(
            "*", f"posexplode({intermediate_col}) as (tile, {self.getOutputCol()})"
        ).drop(intermediate_col)

        return df
