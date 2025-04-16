import os
import csv

import luigi
import pandas as pd
from pyspark.sql import DataFrame

from plantclef.spark import spark_resource


class SubmissionTask(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    dataset_name = luigi.Parameter()
    top_k = luigi.OptionalIntParameter(default=5)
    use_grid = luigi.OptionalBoolParameter(default=False)
    grid_size = luigi.OptionalIntParameter(default=3)
    use_prior = luigi.OptionalBoolParameter(default=False)

    def _get_folder_name(self):
        """Returns the folder name based on parameters."""
        folder_name = f"topk_{self.top_k}"

        if self.use_grid:
            folder_name = f"{folder_name}_grid_{self.grid_size}x{self.grid_size}"

        return folder_name

    def _get_full_output_path(self, with_success=False):
        """Returns the full output path with optional _SUCCESS suffix."""
        folder_name = self._get_folder_name()

        # build the path with prior as a directory if needed
        if self.use_prior:
            path = f"{self.output_path}/{self.dataset_name}/prior/{folder_name}"
        else:
            path = f"{self.output_path}/{self.dataset_name}/{folder_name}"

        if with_success:
            path = f"{path}/_SUCCESS"

        return path

    def output(self):
        # Get path with _SUCCESS marker
        output_path = self._get_full_output_path(with_success=True)
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
            logits = row["logits"]
            top_k_species = self._extract_top_k_species(logits)
            formatted_species = self._format_species_ids(top_k_species)
            records.append({"quadrat_id": image_name, "species_ids": formatted_species})

        pandas_df = pd.DataFrame(records)
        return pandas_df

    def _write_csv_to_pace(self, df):
        """Writes the Pandas DataFrame to a CSV file in GCS."""
        file_name = f"dsgt_run_{self._get_folder_name()}.csv"
        output_path = f"{self._get_full_output_path()}/{file_name}"

        # ensure directory exists before saving
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # write to CSV
        df.to_csv(output_path, sep=",", index=False, quoting=csv.QUOTE_ALL)
        # print submission path
        print(f"Submission path: {output_path}")

    def run(self):
        with spark_resource() as spark:
            # read data
            transformed_df = spark.read.parquet(self.input_path)
            transformed_df = transformed_df.orderBy("image_name")
            transformed_df.printSchema()

            # get prepared dataframe
            pandas_df = self._prepare_and_write_submission(transformed_df)
            self._write_csv_to_pace(pandas_df)

            # write the output
            with self.output().open("w") as f:
                f.write("")
