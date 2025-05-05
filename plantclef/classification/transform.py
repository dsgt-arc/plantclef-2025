import timm
import torch
import pandas as pd
from plantclef.serde import deserialize_image
from plantclef.config import get_class_mappings_file
from plantclef.model_setup import setup_fine_tuned_model
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
from .params import HasModelPath, HasModelName, HasBatchSize


class ClasifierFineTunedDINOv2(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasModelPath,
    HasModelName,
    HasBatchSize,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Wrapper for the fine-tuned DINOv2 model for classification.
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        model_path: str = setup_fine_tuned_model(),
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size: int = 8,
        use_grid: bool = False,
        grid_size: int = 3,
        prior_path: str = None,
    ):
        super().__init__()
        self._setDefault(
            inputCol=input_col,
            outputCol=output_col,
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
        # path for class_mappings.txt file
        self.class_mapping_file = get_class_mappings_file()
        # load class mappings
        self.cid_to_spid = self._load_class_mapping()
        self.use_grid = use_grid
        self.grid_size = grid_size
        self.prior_path = prior_path
        self.prior_df = self._get_prior_df()

    def _load_class_mapping(self):
        with open(self.class_mapping_file) as f:
            class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
        return class_index_to_class_name

    def _get_prior_df(self):
        if self.prior_path:
            return pd.read_parquet(self.prior_path)

    def _get_prior_for_image(self, image_name) -> dict:
        # get prior probabilities for image
        prior_row = self.prior_df[self.prior_df["image_name"] == image_name]
        prior_row = prior_row.iloc[0]["prior_probabilities"]
        return prior_row

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

        # # check on the nvidia stats when generating the predict function
        # self._nvidia_smi()

        def predict(input_data, image_name):
            img = deserialize_image(input_data)  # from bytes to PIL image

            processed_image = self.transforms(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(processed_image)
                probabilities = torch.softmax(logits, dim=1)

                # use Prior probabilities
                if self.prior_path:
                    prior = self._get_prior_for_image(image_name)
                    probabilities = probabilities * torch.tensor(prior).to(self.device)

            return probabilities[0].cpu().numpy().tolist()

        return predict

    def _transform(self, df: DataFrame):
        predict_fn = self._make_predict_fn()
        predict_udf = F.udf(predict_fn, ArrayType(FloatType()))
        return df.withColumn(
            self.getOutputCol(),
            predict_udf(F.col(self.getInputCol()), F.col("image_name")),
        )
