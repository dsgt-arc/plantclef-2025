import luigi
import typer
import logging
from typing_extensions import Annotated
from plantclef.spark import spark_resource


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class ProcessTest(luigi.Task):
    """Task to process embeddings."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    sample_id = luigi.OptionalParameter()

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/metadata/_SUCCESS")

    def run(self):
        try:
            with spark_resource() as spark:
                df = spark.read.parquet(self.input_path)
                df.printSchema()  # This should print, but may be suppressed
                logger.info("Schema printed successfully")
        except Exception as e:
            logger.error(f"Spark job failed: {e}")


class Workflow(luigi.WrapperTask):
    """Workflow with two tasks."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    sample_id = luigi.OptionalParameter()

    def requires(self):
        # either we run a single task or we run all the tasks
        if self.sample_id is not None:
            sample_ids = [self.sample_id]
        else:
            sample_ids = list(range(self.num_tasks))

        logger.info("Creating tasks...")
        logger.info(f"Sample IDs: {sample_ids}")

        tasks = []
        for sample_id in sample_ids:
            logger.info(f"Creating task for sample_id: {sample_id}")
            task = ProcessTest(
                input_path=self.input_path,
                output_path=self.output_path,
                sample_id=sample_id,
            )
            tasks.append(task)
        yield tasks


# Test sbatch
def main(
    input_path: Annotated[str, typer.Argument(help="Input root directory")],
    output_path: Annotated[str, typer.Argument(help="Output root directory")],
    sample_id: Annotated[int, typer.Option(help="Sample ID")] = None,
    scheduler_host: Annotated[str, typer.Option(help="Scheduler host")] = None,
):
    logger.info("Starting workflow execution...")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Sample ID: {sample_id}")

    # run the workflow
    kwargs = {}
    if scheduler_host:
        kwargs["scheduler_host"] = scheduler_host
    else:
        kwargs["local_scheduler"] = True

    logger.info("Calling luigi.build() with parameters...")

    luigi.build(
        [
            Workflow(
                input_path=input_path,
                output_path=output_path,
                sample_id=sample_id,
            )
        ],
        **kwargs,
    )

    logger.info("Workflow execution completed")
