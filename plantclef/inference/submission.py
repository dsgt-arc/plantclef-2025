import csv

import luigi
import pandas as pd
from pyspark.sql import DataFrame

from plantclef.spark import spark_resource


class SubmissionTask(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    top_k = luigi.OptionalIntParameter(default=5)
    use_grid = luigi.OptionalBoolParameter(default=False)
    grid_size = luigi.OptionalIntParameter(default=3)

    def output(self):
        # save the model run
        output_path = f"{self.output_path}/top{self.top_k}_species/_SUCCESS"
        if self.use_grid:
            output_path = f"{self.output_path}/top{self.top_k}_species_grid_{self.grid_size}x{self.grid_size}/_SUCCESS"
        return luigi.LocalTarget(output_path)

    def _format_species_ids(self, species_ids: list) -> str:
        """Formats the species IDs in single square brackets, separated by commas."""
        formatted_ids = ", ".join(str(id) for id in species_ids)
        return f"[{formatted_ids}]"

    def _extract_top_k_species(self, logits: list) -> list:
        """Extracts the top k species from the logits list."""
        top_logits = [list(item.keys())[0] for item in logits[: self.top_k]]
        set_logits = sorted(set(top_logits), key=top_logits.index)
        return set_logits

    def _remove_extension(self, filename: str) -> str:
        """Removes the file extension from the filename."""
        return filename.rsplit(".", 1)[0]

    def _prepare_and_write_submission(self, spark_df: DataFrame) -> DataFrame:
        """Converts Spark DataFrame to Pandas, formats it, and writes to GCS."""
        records = []
        for row in spark_df.collect():
            image_name = self._remove_extension(row["image_name"])
            logits = row["dino_logits"]
            top_k_species = self._extract_top_k_species(logits)
            formatted_species = self._format_species_ids(top_k_species)
            records.append({"plot_id": image_name, "species_ids": formatted_species})

        pandas_df = pd.DataFrame(records)
        return pandas_df

    def _write_csv_to_gcs(self, df):
        """Writes the Pandas DataFrame to a CSV file in GCS."""
        folder_name = f"top_{self.top_k}_species"
        if self.use_grid:
            grid_name = f"grid_{self.grid_size}x{self.grid_size}"
            folder_name = f"{folder_name}_{grid_name}"
        file_name = f"dsgt_run_{folder_name}.csv"
        output_path = f"{self.output_path}/{folder_name}/{file_name}"
        df.to_csv(output_path, sep=";", index=False, quoting=csv.QUOTE_NONE)

    def run(self):
        with spark_resource() as spark:
            # read data
            transformed_df = spark.read.parquet(self.input_path)
            transformed_df = transformed_df.orderBy("image_name")

            # get prepared dataframe
            pandas_df = self._prepare_and_write_submission(transformed_df)
            self._write_csv_to_gcs(pandas_df)

            # write the output
            with self.output().open("w") as f:
                f.write("")
