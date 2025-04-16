import timm
import torch
import pandas as pd
from plantclef.serde import deserialize_image
from plantclef.config import get_class_mappings_file
from plantclef.model_setup import setup_fine_tuned_model
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, MapType, StringType


class HasModelPath(Param):
    """
    Mixin for param model_path: str
    """

    modelPath = Param(
        Params._dummy(),
        "modelPath",
        "The path to the fine-tuned DINOv2 model",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__(
            default=setup_fine_tuned_model(),
            doc="The path to the fine-tuned DINOv2 model",
        )

    def getModelPath(self) -> str:
        return self.getOrDefault(self.modelPath)


class HasModelName(Param):
    """
    Mixin for param model_name: str
    """

    modelName = Param(
        Params._dummy(),
        "modelName",
        "The name of the DINOv2 model to use",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__(
            default="vit_base_patch14_reg4_dinov2.lvd142m",
            doc="The name of the DINOv2 model to use",
        )

    def getModelName(self) -> str:
        return self.getOrDefault(self.modelName)


class HasBatchSize(Param):
    """
    Mixin for param batch_size: int
    """

    batchSize = Param(
        Params._dummy(),
        "batchSize",
        "The batch size to use for embedding extraction",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self):
        super().__init__(
            default=32,
            doc="The batch size to use for embedding extraction",
        )

    def getBatchSize(self) -> int:
        return self.getOrDefault(self.batchSize)


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
        use_prior: bool = False,
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
        self.cluster_df, self.probabilities_df = self._get_cluster_probability_dfs()
        self.use_grid = use_grid
        self.grid_size = grid_size
        self.use_prior = use_prior

    def _load_class_mapping(self):
        with open(self.class_mapping_file) as f:
            class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
        return class_index_to_class_name

    def _get_cluster_probability_dfs(self):
        clustering_path = "~/p-dsgt_clef2025-0/shared/plantclef/data/clustering"
        test_cluster_csv = f"{clustering_path}/test_2025_dominant_clusters.csv"
        test_cluster_probabilities = (
            f"{clustering_path}/test_2025_embed_probabilities_clustered"
        )
        cluster_df = pd.read_csv(test_cluster_csv)
        probabilities_df = pd.read_parquet(test_cluster_probabilities)
        return cluster_df, probabilities_df

    def _get_prior_for_image(self, image_name) -> dict:
        row = self.cluster_df[self.cluster_df["image_name"] == image_name]
        cluster_id = row.iloc[0]["kmeans_cluster"]
        prior_row = self.probabilities_df[
            self.probabilities_df["dominant_cluster"] == cluster_id
        ]
        return prior_row.iloc[0]["renormalized_probabilities"]

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

        def predict(input_data, image_name):
            img = deserialize_image(input_data)  # from bytes to PIL image
            top_k_proba = 10
            limit_logits = 10
            images = [img]
            # use grid to get logits
            if self.use_grid:
                images = self._split_into_grid(img)
            results = []
            for tile in images:
                processed_image = self.transforms(tile).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(processed_image)
                    probabilities = torch.softmax(outputs, dim=1) * 100
                    if self.use_prior:
                        prior = self._get_prior_for_image(image_name)
                        probabilities = probabilities * torch.tensor(prior).to(
                            self.device
                        )
                    top_probs, top_indices = torch.topk(probabilities, k=top_k_proba)
                top_probs = top_probs.cpu().numpy()[0]
                top_indices = top_indices.cpu().numpy()[0]

                result = [
                    {self.cid_to_spid[index]: float(prob)}
                    for index, prob in zip(top_indices, top_probs)
                ]
                results.append(result)
            # flatten the results from all grids, get top probabilities
            flattened_results = [
                item for grid in results for item in grid[:limit_logits]
            ]
            # sort by score in descending order
            sorted_logits = sorted(
                flattened_results, key=lambda x: -list(x.values())[0]
            )
            return sorted_logits

        return predict

    def _transform(self, df: DataFrame):
        predict_fn = self._make_predict_fn()
        predict_udf = F.udf(predict_fn, ArrayType(MapType(StringType(), FloatType())))
        return df.withColumn(
            self.getOutputCol(),
            predict_udf(F.col(self.getInputCol()), F.col("image_name")),
        )
