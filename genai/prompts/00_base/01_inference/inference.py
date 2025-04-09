import os
import json
from google import genai
from google.genai import types  # Use generative_ai for types
from typing import Annotated
import typer

app = typer.Typer()

# --- Configuration ---
# Define the batch of image paths you want to process
# Add or remove image file paths here as needed.
# Example: image_batch = ["image1.jpg", "path/to/image2.png", ...]
IMAGE_BATCH = [
    "CBN-PdlC-A1-20160726.jpg",
    "CBN-Pyr-03-20230706.jpg",
    # Add more image paths here (e.g., up to 10 or your desired batch size)
]
PROMPT_FILE = "PROMPT.md"
SCHEMA_FILE = "schema.json"
# Specify the model capable of handling multiple images and function calling/schema enforcement
# Check Google AI documentation for the latest recommended models (e.g., gemini-1.5-pro)
MODEL_NAME = "models/gemini-2.5-pro-latest"
# --- End Configuration ---


def load_text_file(filepath: str) -> str | None:
    """Loads text content from a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None


def load_json_file(filepath: str) -> dict | None:
    """Loads JSON content from a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None
    except Exception as e:
        print(f"Error reading JSON file {filepath}: {e}")
        return None


def load_image_files() -> list[str]:
    """Loads image files from the specified directory."""
    project_path = "/storage/coda1/p-dsgt_clef2025/0/shared/plantclef"
    image_dir = (
        "data/test_2025/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images"
    )
    image_path = os.path.join(project_path, image_dir)

    # Get all files in directory in their original order
    all_files = os.listdir(image_path)

    # Filter to include only JPG and PNG files
    image_files = [
        os.path.join(image_path, filename)
        for filename in all_files
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    # sort files by name
    image_files.sort()

    return image_files


def load_image_batches(
    all_image_paths: list[str],
    batch_size: int = 10,
) -> list[list[str]]:
    """
    Loads image files from the specified directory and returns them in batches.

    Args:
        all_image_paths: A list of paths to the image files.
        batch_size: The number of images to include in each batch.
    Returns:
        A list of lists, where each inner list contains paths to batch_size images.
    """
    # split into batches
    batches = []
    for i in range(0, len(all_image_paths), batch_size):
        batch = all_image_paths[i : i + batch_size]
        batches.append(batch)
    print(f"Created {len(batches)} batches of images (batch size: {batch_size})")

    return batches


def generate_botanical_analysis(
    image_paths: list[str],
    prompt_template: str,
    output_schema_dict: dict,
    model_name: str = MODEL_NAME,
) -> str | None:
    """
    Generates botanical analysis for a batch of images using Google Generative AI.

    Args:
        image_paths: A list of paths to the image files.
        prompt_template: The full system prompt template string.
        output_schema_dict: The dictionary representing the desired JSON output schema.
        model_name: The name of the Gemini model to use.

    Returns:
        The generated JSON string response from the model, or None if an error occurs.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return None

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Google GenAI Client: {e}")
        return None

    uploaded_files = []
    valid_image_paths = []
    print("Uploading files...")
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"   Warning: File not found at {img_path}. Skipping.")
            continue
        print(f" - Uploading {img_path}...")
        # Upload the file
        uploaded_file = client.files.upload(path=img_path)
        uploaded_files.append(uploaded_file)
        valid_image_paths.append(img_path)  # Keep track of successfully uploaded paths
        print(f"   - Uploaded {img_path} as {uploaded_file.name} ({uploaded_file.uri})")

    if not uploaded_files:
        print("Error: No valid image files were provided or uploaded.")
        return None

    # --- Prepare Prompt Components ---
    # 1. System Instruction (from the prompt file)
    system_instruction = types.Part.from_text(prompt_template)

    # 2. User Request Parts (Images + Filenames text)
    user_prompt_parts = []
    # Add image parts first
    for f in uploaded_files:
        user_prompt_parts.append(
            types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type)
        )

    # Add the text part listing the original image filenames
    # Use the filenames from valid_image_paths which correspond to uploaded_files
    image_names = [os.path.basename(p) for p in valid_image_paths]
    print(f"Image names: {image_names}")
    user_prompt_parts.append(
        types.Part.from_text(f"image names: {', '.join(image_names)}")
    )

    # Construct the final contents list for the API call
    # The user prompt only contains the images and the filename list.
    # The main instructions are in the system_instruction.
    contents = [types.Content(role="user", parts=user_prompt_parts)]

    # 3. Output Schema
    output_schema = types.Schema.from_dict(output_schema_dict)

    # 4. Generation Configuration
    generation_config = types.GenerationConfig(
        # Specify the response MIME type for JSON output.
        response_mime_type="application/json",
        # Pass the schema for the desired output format.
        response_schema=output_schema,
        # You might adjust temperature, top_k, top_p etc. if needed
        # temperature=0.5,
    )

    # --- Call the Model ---
    print(f"\nGenerating content using model '{model_name}'...")
    full_response = ""
    # Make the API request with system instruction, user content, and generation config
    response_stream = client.generative_model(model_name=model_name).generate_content(
        contents=contents,
        generation_config=generation_config,
        system_instruction=system_instruction,
        stream=True,
    )

    # Print the streamed response chunks
    for chunk in response_stream:
        if hasattr(chunk, "text"):
            print(chunk.text, end="")
            full_response += chunk.text
        elif hasattr(chunk, "parts"):  # Handle potential non-text parts if necessary
            pass  # Or log/process them

    print("\n--- Generation Complete ---")

    return full_response


@app.command()
def process(
    model_name: Annotated[
        str, typer.Option(help="Model name to use.")
    ] = "models/gemini-2.5-pro-latest",
    prompt_file: Annotated[
        str, typer.Option(help="Path to the prompt file.")
    ] = "PROMPT.md",
    schema_file: Annotated[
        str, typer.Option(help="Path to the schema file.")
    ] = "schema.json",
    batch_size: Annotated[
        int, typer.Option(help="Number of images to process in a batch.")
    ] = 10,
):
    print("--- Starting PlantCLEF Batch Analysis ---")

    # 1. Load Prompt Template
    print(f"Loading prompt from: {prompt_file}")
    prompt = load_text_file(prompt_file)
    print(f"Prompt loaded: {prompt[:50]}...")  # Print first 50 chars for brevity

    # 2. Load Output Schema
    print(f"Loading schema from: {schema_file}")
    schema = load_json_file(schema_file)
    print(f"Schema loaded: {schema}")  # Print the schema for verification

    # 3. Load Image Files
    print("Loading image files...")
    image_files = load_image_files()
    print(f"Loaded {len(image_files)} image files.")
    print(f"Image files: {image_files[:10]}")

    # 4. Create Batches
    image_batches = load_image_batches(image_files, batch_size=batch_size)
    print(f"Created {len(image_batches)} batches of images.")
    print(f"First batch: {image_batches[0]}")

    # # 4. Run Generation
    # generated_json = generate_botanical_analysis(
    #     image_paths=IMAGE_BATCH, prompt_template=prompt, output_schema_dict=schema
    # )

    # # 5. (Optional) Post-processing / Validation
    # if generated_json:
    #     print("\n--- Final Generated JSON Output ---")
    #     # The output is already printed chunk by chunk, but we print the final accumulated one again
    #     # for clarity, or you could parse/validate it here.
    #     print("Final JSON output:")
    #     print(generated_json)


if __name__ == "__main__":
    app()
