import functools
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated
import time

import tqdm
import typer
from google import genai
from google.genai import types
import json

app = typer.Typer()


def generate(client, input_text: str, model="gemini-2.0-flash") -> str:
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=input_text)])
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.ARRAY,
            description="List of structured outputs for initial species filtering, using the 'nullable' keyword (JSON Schema Draft 2020-12 style).",
            items=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                description="Schema for a single species entry.",
                required=[
                    "species_id",
                    "scientific_name",
                    "filter_is_vascular",
                    "filter_is_quadrat_relevant",
                    "filter_is_in_europe",
                    "filter_is_in_pyrenees_med",
                ],
                properties={
                    "species_id": genai.types.Schema(
                        type=genai.types.Type.INTEGER,
                        description="Unique identifier for the species, from the input data.",
                    ),
                    "scientific_name": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Scientific name of the species.",
                    ),
                    "filter_is_vascular": genai.types.Schema(
                        type=genai.types.Type.BOOLEAN,
                        description="Is the plant determined to be vascular? (True=Yes, False=No, Null=Uncertain/NotFound).",
                        nullable="True",
                    ),
                    "filter_is_quadrat_relevant": genai.types.Schema(
                        type=genai.types.Type.BOOLEAN,
                        description="Is the plant determined to be relevant for a quadrat based on habit/size? (True=Yes, False=No, Null=Uncertain/NotFound).",
                        nullable="True",
                    ),
                    "filter_is_in_europe": genai.types.Schema(
                        type=genai.types.Type.BOOLEAN,
                        description="Is the plant determined to be present in Europe? (True=Yes, False=No, Null=Uncertain/NotFound).",
                        nullable="True",
                    ),
                    "filter_is_in_pyrenees_med": genai.types.Schema(
                        type=genai.types.Type.BOOLEAN,
                        description="Is the plant determined to be present in the Pyrenees or Med. Basin? (True=Yes, False=No, Null=Uncertain/NotFound).",
                        nullable="True",
                    ),
                    "justification": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Brief justification or notes supporting the filter determinations.",
                        nullable="True",
                    ),
                },
            ),
        ),
        system_instruction=[
            types.Part.from_text(
                text=(Path(__file__).parent / "PROMPT_STRUCTURE.md").read_text()
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


def validate_output(response: str) -> bool:
    """Ensure the resulting document is valid json."""
    try:
        json.loads(response)
        return True
    except json.JSONDecodeError:
        return False


def process_shard(input_file_path, output_dir_path, api_key):
    """Process a single shard of data."""
    # take the input file and write it as json, in a sibling subdirectory
    input_file_path = Path(input_file_path)
    output_dir_path = Path(output_dir_path)

    output_file_path = output_dir_path / f"structure/{input_file_path.stem}.json"
    if output_file_path.exists():
        return f"Skipped {input_file_path} (file already exists)"

    # Create a client in the worker process
    client = genai.Client(api_key=api_key)

    # Generate the text
    input_text = Path(input_file_path).read_text()
    try:
        resp_text = generate(client, input_text)
    except Exception as e:
        print(f"Error generating response for shard {input_file_path}: {e}")
        return f"Failed for shard {input_file_path}"

    if not (validates := validate_output(response=resp_text)):
        print(f"Validation failed for shard {output_file_path}")
        # write the output with an extension to indicate failure
        # this should get a timestamp

        output_file_path = output_file_path.with_suffix(
            f".{int(time.time())}.failed.json"
        )

    # Write output
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.write_text(resp_text)
    if validates:
        return f"Processed shard {input_file_path}"
    else:
        return f"Failed for shard {input_file_path}"


@app.command()
def process(
    input_path: Annotated[str, typer.Argument(help="Path to the input file")],
    limit: Annotated[
        int, typer.Option(help="Limit the number of lines to process")
    ] = -1,
    workers: Annotated[int, typer.Option(help="Number of worker processes")] = 4,
):
    """Take a yaml data and process them into structured json."""
    # Create list of shard indices to process
    paths = sorted(Path(input_path).glob("info/part-*.md"))
    if not paths:
        raise FileNotFoundError(f"No input files found in {input_path}")
    if limit > 0:
        paths = paths[:limit]

    # Create a partial function with fixed arguments
    worker_func = functools.partial(
        process_shard,
        output_dir_path=Path(input_path),
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # Process using multiprocessing pool with tqdm
    with Pool(processes=workers) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(worker_func, paths),
                total=len(paths),
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
