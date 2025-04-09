import base64
import os
from google import genai
from google.genai import types
from pathlib import Path
import typer
import pandas as pd
import tqdm

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


@app.command()
def process(
    input_path: str = typer.Argument(..., help="Path to the input file"),
    output_path: str = typer.Argument(..., help="Path to the output directory"),
    shards: int = typer.Option(1, help="Number of shards to split the input file into"),
    limit: int = typer.Option(-1, help="Limit the number of lines to process"),
):
    """Take a bunch of species csv data and find information about them.

    species_id,species,genus,family
    1355868,Lactuca virosa L.,Lactuca,Asteraceae
    1355869,Crepis capillaris (L.) Wallr.,Crepis,Asteraceae
    1355870,Crepis foetida L.,Crepis,Asteraceae
    1355871,Hypochaeris glabra L.,Hypochaeris,Asteraceae
    """
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    input_file_path = Path(input_path)
    output_dir_path = Path(output_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file_path, header=True)
    # now we shard the input data by taking the mod of the total number of shards

    # TODO: parallellize the process
    for i in range(shards):
        if limit > 0 and i >= limit:
            break
        subset = df[df.species_id % shards == i]
        print(subset)


if __name__ == "__main__":
    process()
