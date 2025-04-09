import functools
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated
import time

import pandas as pd
import tqdm
import typer
from google import genai
from google.genai import types

app = typer.Typer()


def generate(client, input_text: str, model="gemini-2.0-flash") -> str:
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=input_text)])
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(
                text=(Path(__file__).parent / "PROMPT_INFO.md").read_text()
            )
        ],
    )

    resp_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        resp_text += chunk.text

    return resp_text


def validate_output(df: pd.DataFrame, response: str) -> bool:
    """Ensure that all species ids that were requested are present in the response."""
    species_ids = set(df["species_id"].astype(str))
    response_ids = set()

    # just check if the species id string appears in the response
    for species_id in species_ids:
        if str(species_id) in response:
            response_ids.add(str(species_id))

    return species_ids == response_ids


def process_shard(i, input_file_path, output_dir_path, shards, api_key):
    """Process a single shard of data."""
    # Check if output file exists
    output_file_path = output_dir_path / f"info/part-{i:03d}.md"
    if output_file_path.exists():
        return f"Skipped shard {i} (file already exists)"

    # Create a client in the worker process
    client = genai.Client(api_key=api_key)

    # Read and filter data
    df = pd.read_csv(input_file_path)
    subset = df[df.species_id % shards == i]

    # Generate the text
    input_text = (
        "Here is the species data in csv format with headers"
        f" and {len(subset)} rows:\n"
    )
    input_text = subset.to_csv(index=False)
    try:
        resp_text = generate(client, input_text)
    except Exception as e:
        print(f"Error generating response for shard {i}: {e}")
        return f"Failed for shard {i}"

    if not (validates := validate_output(df=subset, response=resp_text)):
        print(f"Validation failed for shard {output_file_path}")
        # write the output with an extension to indicate failure
        # this should get a timestamp

        output_file_path = output_file_path.with_suffix(
            f".{int(time.time())}.failed.md"
        )

    # Write output
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.write_text(resp_text)
    if validates:
        return f"Processed shard {i}"
    else:
        return f"Failed for shard {i}"


@app.command()
def process(
    input_path: Annotated[str, typer.Argument(help="Path to the input file")],
    output_path: Annotated[str, typer.Argument(help="Path to the output directory")],
    shards: Annotated[
        int, typer.Option(help="Number of shards to split the input file into")
    ] = 200,
    limit: Annotated[
        int, typer.Option(help="Limit the number of lines to process")
    ] = -1,
    workers: Annotated[int, typer.Option(help="Number of worker processes")] = 4,
):
    """Take a bunch of species csv data and find information about them.

    species_id,species,genus,family
    1355868,Lactuca virosa L.,Lactuca,Asteraceae
    1355869,Crepis capillaris (L.) Wallr.,Crepis,Asteraceae
    1355870,Crepis foetida L.,Crepis,Asteraceae
    1355871,Hypochaeris glabra L.,Hypochaeris,Asteraceae
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    input_file_path = Path(input_path)
    output_dir_path = Path(output_path)

    # Create list of shard indices to process
    shard_indices = list(range(shards))
    if limit > 0:
        shard_indices = shard_indices[:limit]

    # Create a partial function with fixed arguments
    worker_func = functools.partial(
        process_shard,
        input_file_path=input_file_path,
        output_dir_path=output_dir_path,
        shards=shards,
        api_key=api_key,
    )

    # Process using multiprocessing pool with tqdm
    with Pool(processes=workers) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(worker_func, shard_indices),
                total=len(shard_indices),
                desc="Processing shards",
            )
        )

    # Print summary
    skipped = [r for r in results if "Skipped" in r]
    failed = [r for r in results if "Failed" in r]
    success = [len(results) - len(skipped) - len(failed)]
    print(f"Success: {success[0]}, Failed: {len(failed)}, Skipped: {len(skipped)}")


if __name__ == "__main__":
    app()
