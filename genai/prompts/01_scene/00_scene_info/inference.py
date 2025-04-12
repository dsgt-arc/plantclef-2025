import functools
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated, Any, List

import tqdm
import typer
from dotenv import load_dotenv
from google import genai
from google.genai import types

app = typer.Typer()


def generate(
    client,
    image_paths: list[Path],
    model_name: str = "gemini-2.0-flash-thinking-exp-01-21",
) -> str:
    files = [client.files.upload(file=p) for p in image_paths]
    file_parts = [
        types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type)
        for file in files
    ]
    image_parts = [
        types.Part.from_text(
            text=(
                "Now, analyze the following batch of images and their corresponding filenames\n"
                "image names: " + json.dumps([p.name for p in image_paths])
            )
        )
    ]
    contents = [types.Content(role="user", parts=file_parts + image_parts)]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(
                text=((Path(__file__).parent / "PROMPT.md").read_text())
            )
        ],
    )

    text = ""
    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=generate_content_config,
    ):
        if not chunk.text:
            print("Empty chunk received")
            continue
        text += chunk.text
    return text


def generate_batches(
    items: list[Any],
    batch_size: int = 10,
) -> list[list[Path]]:
    """Batch items in a list into smaller lists."""
    batches = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batches.append(batch)
    return batches


def validate_output(image_paths: List[Path], response: str) -> bool:
    """Ensure that all images that were processed are present in the response."""
    try:
        for p in image_paths:
            assert p.name in response
    except Exception:
        return False
    return True


def process_batch(
    batch_index: int,
    image_batches: List[List[Path]],
    output_dir_path: Path,
    model_name: str,
    api_key: str,
) -> str:
    """Process a single batch of images."""
    # Check if output file exists
    output_file_path = output_dir_path / f"results/batch-{batch_index:03d}.md"
    if output_file_path.exists():
        return f"Skipped batch {batch_index} (file already exists)"

    # Create a client in the worker process
    client = genai.Client(api_key=api_key)

    # Get the batch to process
    image_paths = image_batches[batch_index]

    try:
        resp_text = generate(
            client=client,
            image_paths=image_paths,
            model_name=model_name,
        )
    except Exception as e:
        print(f"Error generating response for batch {batch_index}: {e}")
        return f"Failed for batch {batch_index}"

    # Validate output
    if not (validates := validate_output(image_paths=image_paths, response=resp_text)):
        print(f"Validation failed for batch {output_file_path}")
        # Write the output with an extension to indicate failure
        output_file_path = output_file_path.with_suffix(
            f".{int(time.time())}.failed.md"
        )

    # Write output
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.write_text(resp_text)

    if validates:
        return f"Processed batch {batch_index}"
    else:
        return f"Failed for batch {batch_index}"


@app.command()
def process(
    image_path: Annotated[str, typer.Argument(help="Path to the images directory")],
    output_path: Annotated[str, typer.Argument(help="Path to the output directory")],
    model_name: Annotated[
        str, typer.Option(help="Model name to use")
    ] = "gemini-2.0-flash-thinking-exp-01-21",
    batch_size: Annotated[
        int, typer.Option(help="Number of images to process in a batch")
    ] = 4,
    limit: Annotated[
        int, typer.Option(help="Limit the number of batches to process")
    ] = -1,
    workers: Annotated[int, typer.Option(help="Number of worker processes")] = 4,
):
    """Process batches of images through the model and save results as JSON."""
    load_dotenv()
    api_key = os.environ["GEMINI_API_KEY"]

    output_dir_path = Path(output_path)

    # Load and batch images
    paths = sorted(Path(image_path).glob("*"))
    batches = generate_batches(paths, batch_size)

    # Create list of batch indices to process
    batch_indices = list(range(len(batches)))
    if limit > 0:
        batch_indices = batch_indices[:limit]

    # Create a partial function with fixed arguments
    worker_func = functools.partial(
        process_batch,
        image_batches=batches,
        output_dir_path=output_dir_path,
        model_name=model_name,
        api_key=api_key,
    )

    # Process using multiprocessing pool with tqdm
    with Pool(processes=workers) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(worker_func, batch_indices),
                total=len(batch_indices),
                desc="Processing image batches",
            )
        )

    # Print summary
    skipped = [r for r in results if "Skipped" in r]
    failed = [r for r in results if "Failed" in r]
    succeeded = len(results) - len(skipped) - len(failed)
    print(f"Success: {succeeded}, Failed: {len(failed)}, Skipped: {len(skipped)}")


if __name__ == "__main__":
    app()
