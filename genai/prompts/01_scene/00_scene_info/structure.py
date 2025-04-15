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
import dotenv

dotenv.load_dotenv()

app = typer.Typer()


def generate(client, input_text: str, model="gemini-2.0-flash") -> str:
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=input_text)])
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.ARRAY,
            items=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=[
                    "image_filename",
                    "image_capture_date",
                    "derived_season",
                    "metadata_inferred_region",
                    "metadata_inferred_ecoregion",
                    "estimated_region_visual",
                    "habitat_predicted_types",
                    "habitat_reasoning",
                    "habitat_substrate_appearance",
                    "cover_veg_pct",
                    "cover_ground_pct",
                    "cover_rock_pct",
                    "cover_litter_pct",
                    "cover_reasoning",
                    "non_plant_elements",
                    "veg_dominant_forms",
                    "veg_dominant_forms_reasoning",
                    "veg_height_cm",
                    "veg_height_reasoning",
                    "veg_canopy_structure",
                    "veg_canopy_reasoning",
                    "veg_woodiness_observed",
                    "visual_phenology_notes",
                    "trait_leaf_types",
                    "trait_flower_colors",
                    "trait_flower_shapes",
                    "trait_other_features",
                    "diversity_estimated_richness",
                    "diversity_richness_confidence",
                    "diversity_richness_reasoning",
                    "summary_overall",
                    "transcription_changes",
                ],
                properties={
                    "image_filename": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="The filename of the specific image being analyzed.",
                    ),
                    "image_capture_date": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Date the image was captured, formatted as 'YYYY-MM-DD'.",
                    ),
                    "derived_season": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Season inferred from the image capture date.",
                        enum=["Spring", "Summer", "Autumn", "Winter"],
                    ),
                    "metadata_inferred_region": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Geographic region hint derived from filename or metadata.",
                    ),
                    "metadata_inferred_ecoregion": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Ecoregion hint derived from filename or metadata.",
                    ),
                    "estimated_region_visual": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Visual description of the environment/region suggested by this image.",
                    ),
                    "habitat_predicted_types": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        description="List of specific habitat types predicted based on visual cues.",
                        items=genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    ),
                    "habitat_reasoning": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Explanation of the visual evidence supporting the habitat prediction.",
                    ),
                    "habitat_substrate_appearance": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        description="Descriptions of the ground substrate visible in the image.",
                        items=genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    ),
                    "cover_veg_pct": genai.types.Schema(
                        type=genai.types.Type.NUMBER,
                        description="Estimated percentage of area covered by vegetation.",
                    ),
                    "cover_ground_pct": genai.types.Schema(
                        type=genai.types.Type.NUMBER,
                        description="Estimated percentage of area covered by bare soil or ground.",
                    ),
                    "cover_rock_pct": genai.types.Schema(
                        type=genai.types.Type.NUMBER,
                        description="Estimated percentage of area covered by rocks or gravel.",
                    ),
                    "cover_litter_pct": genai.types.Schema(
                        type=genai.types.Type.NUMBER,
                        description="Estimated percentage of area covered by litter (e.g., leaves, detritus).",
                    ),
                    "cover_reasoning": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Justification for the cover estimates based on image content.",
                    ),
                    "non_plant_elements": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        description="List of visible non-plant items in the image.",
                        items=genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    ),
                    "veg_dominant_forms": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        description="Most visually dominant plant growth forms.",
                        items=genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    ),
                    "veg_dominant_forms_reasoning": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Explanation of the dominance call for the vegetation forms.",
                    ),
                    "veg_height_cm": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Estimated height range of vegetation, e.g., '< 15cm'.",
                    ),
                    "veg_height_reasoning": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Reasoning for the estimated vegetation height.",
                    ),
                    "veg_canopy_structure": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Description of the canopy structure, e.g., 'Open / Patchy'.",
                    ),
                    "veg_canopy_reasoning": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Explanation for the canopy structure observed.",
                    ),
                    "veg_woodiness_observed": genai.types.Schema(
                        type=genai.types.Type.BOOLEAN,
                        description="Indicates whether woody vegetation is observed.",
                    ),
                    "visual_phenology_notes": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Notes on seasonal or reproductive stage of visible plants.",
                    ),
                    "trait_leaf_types": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        description="Types of leaf morphology observed.",
                        items=genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    ),
                    "trait_flower_colors": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        description="Colors of flowers observed.",
                        items=genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    ),
                    "trait_flower_shapes": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        description="Shapes of flowers observed.",
                        items=genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    ),
                    "trait_other_features": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        description="Other notable morphological features.",
                        items=genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    ),
                    "diversity_estimated_richness": genai.types.Schema(
                        type=genai.types.Type.NUMBER,
                        description="Number of distinct plant types/species observed.",
                    ),
                    "diversity_richness_confidence": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Confidence level in the estimated richness.",
                        enum=["Low", "Medium", "High"],
                    ),
                    "diversity_richness_reasoning": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Rationale for the estimated richness.",
                    ),
                    "taxa_suggested_groups": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        description="Broad taxonomic groups inferred from visual evidence.",
                        items=genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    ),
                    "taxa_groups_reasoning": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Reasoning for suggested taxonomic group(s).",
                    ),
                    "summary_overall": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Overall synthesis of scene features, structure, habitat, and dominant visual traits.",
                    ),
                    "transcription_changes": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Any changes made to the transcription of the image filename or metadata.",
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


def validate_output(input_text: str, response: str) -> bool:
    """Ensure the resulting document is valid json."""
    try:
        # image_filename: 2024-CEV3-20240602.jpg
        input_lines = input_text.splitlines()
        input_filenames = [
            line.split(": ")[1].strip()
            for line in input_lines
            if line.startswith("image_filename:")
        ]
        data = json.loads(response)
        response_filenames = [item["image_filename"] for item in data]
        # now check if the filenames in the input text exist in the response
        # check the sets are the same
        if set(input_filenames) != set(response_filenames):
            return False
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

    if not (validates := validate_output(input_text=input_text, response=resp_text)):
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
    paths = sorted(Path(input_path).glob("results/*.md"))
    if not paths:
        raise FileNotFoundError(f"No input files found in {input_path}")
    if limit > 0:
        paths = paths[:limit]

    print(f"Found {len(paths)} shards to process.")

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
