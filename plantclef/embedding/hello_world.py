import typer
import logging
from typing_extensions import Annotated

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    typer.run(main)
