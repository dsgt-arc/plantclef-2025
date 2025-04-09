import os
import typer
from typing import Annotated
from dotenv import load_dotenv
from google import genai
from google.genai import types  # Use generative_ai for types

app = typer.Typer()


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


def load_image_files() -> list[str]:
    """Loads image files from the specified directory and returns string paths."""
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


def generate(
    client,
    image_paths: list[str],
    prompt_template: str,
    model_name: str = "gemini-2.0-flash",
) -> str:
    # get image_names from image_paths
    image_names = [img.split("images/")[-1] for img in image_paths]

    # upload files to client
    files = [client.files.upload(file=img_path) for img_path in image_paths]

    # create a list of parts for each uploaded file
    file_parts = [
        types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type)
        for file in files
    ]
    image_parts = [
        types.Part.from_text(text=f"image names: {img_name}")
        for img_name in image_names
    ]

    # add the text prompt as another part
    contents = [
        types.Content(
            role="user",
            parts=file_parts + image_parts,
        ),
    ]

    # generate content
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.ARRAY,
            description="An array containing the analysis results for each image in a batch.",
            items=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=["image_name", "scene_description", "identified_species"],
                properties={
                    "image_name": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="The filename of the analyzed image.",
                    ),
                    "scene_description": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="A brief summary of the quadrat's appearance for a single image, including vegetation type, cover, substrate, and notable features (under 100 words).",
                    ),
                    "identified_species": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        description="A list of unique scientific names (Genus species or Genus sp.) of plant species identified in that specific image.",
                        items=genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="Scientific name of an identified plant species.",
                            enum=[
                                "Medicago truncatula Gaertn.",
                                "Sesleria albicans Kit. ex Schult.",
                                "Primula halleri J.F.Gmel.",
                                "Viola montcaunica Pau",
                                "Bromus hordeaceus L.",
                                "Pyrola media Sw.",
                                "Carex aterrima Hoppe",
                                "Quercus × rosacea Bechst.",
                                "Trifolium physodes Steven ex M.Bieb.",
                                "Carex mucronata All.",
                                "Valeriana celtica L.",
                                "Dianthus benearnensis Loret",
                                "Carex atrata L.",
                                "Luzula spicata (L.) DC.",
                                "Carex arenaria L.",
                                "Primula pedemontana E.Thomas ex Gaudin",
                                "Lotus arenarius Brot.",
                                "Gentiana pumila Jacq.",
                                "Phyteuma globulariifolium Sternb. & Hoppe",
                                "Volutaria tubuliflora (Murb.) Sennen",
                                "Alchemilla pentaphyllea L.",
                                "Phleum phleoides (L.) H.Karst.",
                                "Anthoxanthum nipponicum Honda",
                                "Taeniatherum caput-medusae (L.) Nevski",
                                "Convolvulus farinosus L.",
                                "Helichrysum saxatile Moris",
                                "Bromus tectorum L.",
                                "Rhododendron ponticum L.",
                                "Veronica elliptica G.Forst.",
                                "Chamorchis alpina (L.) Rich.",
                                "Saxifraga adscendens L.",
                                "Atriplex semibaccata R.Br.",
                                "Atriplex portulacoides L.",
                                "Pedicularis oederi Vahl",
                                "Salix bicolor Ehrh. ex Willd.",
                                "Psilathera ovata (Hoppe) Deyl",
                                "Oreochloa disticha (Wulfen) Link",
                                "Lotus alpinus (Ser.) Schleich. ex Ramond",
                                "Gagea serotina (L.) Ker Gawl.",
                                "Sibbaldia procumbens L.",
                                "Gentiana terglouensis Hacq.",
                                "Hordeum secalinum Schreb.",
                                "Viola pyrenaica Ramond ex DC.",
                                "Crocus nevadensis Amo & Campo",
                                "Saxifraga androsacea L.",
                                "Saxifraga cernua L.",
                                "Carex capitata Sol.",
                                "Gentiana verna L.",
                                "Anthemis secundiramea Biv.",
                                "Ludwigia repens J.R.Forst.",
                                "Poa chaixii Vill.",
                                "Crocus carpetanus Boiss. & Reut.",
                                "Quercus coccifera L.",
                                "Trifolium spadiceum L.",
                                "Posidonia oceanica (L.) Delile",
                                "Bromus madritensis L.",
                                "Marcus-kochia littorea (L.) Al-Shehbaz",
                                "Daucus pumilus (L.) Hoffmanns. & Link",
                                "Trifolium occidentale Coombe",
                                "Androsace adfinis Biroli",
                                "Viola nummulariifolia Vill.",
                                "Lotus tenuis Waldst. & Kit. ex Willd.",
                                "Oreochloa elegans (Sennen) A.W.Hill",
                                "Bromus inermis Leyss.",
                                "Carex glacialis Mack.",
                                "Saxifraga florulenta Moretti",
                                "Molinia caerulea (L.) Moench",
                                "Anabasis articulata (Forssk.) Moq.",
                                "Thinopyrum junceum (L.) Á.Löve",
                                "Rhododendron × intermedium Tausch",
                                "Lygeum spartum Loefl. ex L.",
                                "Androsace argentea (C.F.Gaertn.) Lapeyr.",
                                "Daphne oleoides Schreb.",
                                "Festuca guestfalica Boenn. ex Rchb.",
                                "Isoetes histrix Bory",
                                "Quercus rotundifolia Lam.",
                                "Calamagrostis arenaria (L.) Roth",
                                "Epilobium anagallidifolium Lam.",
                                "Utricularia subulata L.",
                                "Salicornia fruticosa (L.) L.",
                                "Puccinellia maritima (Huds.) Parl.",
                                "Blysmus compressus (L.) Panz. ex Link",
                                "Cochlearia pyrenaica DC.",
                                "Ranunculus nigrescens Freyn",
                                "Omalotheca supina (L.) DC.",
                                "Poa supina Schrad.",
                                "Scrophularia frutescens L.",
                                "Malva nicaeensis All.",
                                "Tephroseris integrifolia (L.) Holub",
                                "Bromus rubens L.",
                                "Helichrysum stoechas (L.) Moench",
                                "Geum montanum L.",
                                "Helictochloa pratensis (L.) Romero Zarco",
                                "Myosotis secunda Al.Murray",
                                "Thalictrum alpinum L.",
                                "Gymnadenia corneliana (Beauverd) Teppner & E.Klein",
                                "Veronica repens Clarion ex DC.",
                                "Helictotrichon parlatorei (J.Woods) Pilg.",
                                "Calystegia soldanella (L.) R.Br.",
                                "Agropyron cristatum (L.) Gaertn.",
                                "Polygonum arenastrum Boreau",
                                "Bromus sterilis L.",
                                "Marcus-kochia triloba (L.) Al-Shehbaz",
                                "Plantago atrata Hoppe",
                                "Phalaris minor Retz.",
                                "Gentiana utriculosa L.",
                                "Narcissus cyclamineus DC.",
                                "Sporobolus pungens (Schreb.) Kunth",
                                "Rumex spinosus L.",
                                "Medicago marina L.",
                                "Viola crassiuscula Bory",
                                "Festuca indigesta Boiss.",
                                "Helictochloa versicolor (Vill.) Romero Zarco",
                                "Carex myosuroides Vill.",
                                "Logfia minima (Sm.) Dumort.",
                                "Lotus glacialis (Boiss.) Pau",
                                "Salix herbacea L.",
                                "Anthemis maritima L.",
                                "Asplenium cuneifolium Viv.",
                                "Pinguicula nevadensis (H.Lindb.) Casper",
                                "Elymus caninus (L.) L.",
                                "Carex maritima Gunnerus",
                                "Antennaria dioica (L.) Gaertn.",
                                "Rhamnus pumila Turra",
                                "Quercus petraea (Matt.) Liebl.",
                                "Fritillaria montana Hoppe ex W.D.J.Koch",
                                "Carex curvula All.",
                                "Alopecurus gerardii (All.) Vill.",
                                "Luronium natans (L.) Raf.",
                                "Trifolium thalii Vill.",
                                "Antennaria carpatica (Wahlenb.) Bluff & Fingerh.",
                                "Leontopodium nivale (Ten.) A.Huet ex Hand.-Mazz.",
                                "Lotus creticus L.",
                                "Soldanella alpina L.",
                                "Salicornia perennans Willd.",
                                "Eleusine tristachya (Lam.) Lam.",
                                "Phleum bertolonii DC.",
                                "Helichrysum arenarium (L.) Moench",
                                "Luzula lutea (All.) DC.",
                                "Dianthus langeanus Willk.",
                                "Arthrocaulon macrostachyum (Moric.) Piirainen & G.Kadereit",
                                "Festuca nigrescens Lam.",
                                "Oxytropis lapponica (Wahlenb.) J.Gay",
                                "Veronica bellidioides L.",
                                "Bacopa rotundifolia (Michx.) Wettst.",
                                "Trifolium alpinum L.",
                                "Carex sempervirens Vill.",
                                "Botrychium simplex E.Hitchc.",
                                "Arenaria biflora L.",
                                "Festuca violacea Ser. ex Gaudin",
                                "Luzula alpinopilosa (Chaix) Breistr.",
                                "Carex umbrosa Host",
                                "Biarum tenuifolium (L.) Schott",
                                "Salicornia europaea L.",
                                "Primula integrifolia L.",
                                "Ilex aquifolium L.",
                                "Myoporum insulare R.Br.",
                                "Narcissus minor L.",
                                "Saxifraga hirculus L.",
                                "Pinguicula corsica Bernard & Gren.",
                                "Salix pyrenaica Gouan",
                                "Salicornia perennis Mill.",
                                "Cardamine asarifolia L.",
                                "Rhododendron ferrugineum L.",
                                "Scorzoneroides helvetica (Mérat) Holub",
                                "Cyperus capitatus Vand.",
                                "Gentiana clusii Perr. & Songeon",
                                "Poa cenisia All.",
                                "Medicago littoralis Rohde ex Loisel.",
                                "Salix serpillifolia Scop.",
                                "Omalotheca hoppeana (W.D.J.Koch) Sch.Bip. & F.W.Schultz",
                                "Pteridium aquilinum (L.) Kuhn",
                                "Chamaemespilus alpina (Mill.) K.R.Robertson & J.B.Phipps",
                                "Armeria ruscinonensis Girard",
                                "Helichrysum italicum (Roth) G.Don",
                                "Veronica nevadensis (Pau) Pau",
                                "Homogyne alpina (L.) Cass.",
                                "Hordeum marinum Huds.",
                                "Geum pyrenaicum Mill.",
                                "Matthiola sinuata (L.) W.T.Aiton",
                                "Callianthemum coriandrifolium Rchb.",
                                "Narcissus cuneiflorus (Salisb. ex Haw.) Link",
                                "Gentiana alpina Vill.",
                                "Festuca ovina L.",
                                "Agrostis gigantea Roth",
                                "Geum reptans L.",
                                "Micranthes hieraciifolia (Waldst. & Kit. ex Willd.) Haw.",
                                "Luzula alpina Hoppe",
                                "Calamagrostis arundinacea (L.) Roth",
                                "Carex lachenalii Schkuhr",
                                "Crocus corsicus Vanucchi",
                                "Molinia arundinacea Schrank",
                                "Carex firma Host",
                                "Rhamnus alaternus L.",
                                "Festuca quadriflora Honck.",
                                "Campanula herminii Hoffmanns. & Link",
                                "Luzula pediformis (Chaix) DC.",
                                "Sempervivum minutum (Kunze ex Willk.) Nyman ex Pau",
                                "Gentiana ligustica R.Vilm. & Chopinet",
                                "Carex pyrenaica Wahlenb.",
                                "Salix retusa L.",
                                "Carex parviflora Host",
                                "Comastoma tenellum (Rottb.) Toyok.",
                            ],
                        ),
                    ),
                },
            ),
        ),
        system_instruction=[types.Part.from_text(text=prompt_template)],
    )

    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")


@app.command()
def process(
    model_name: Annotated[
        str, typer.Option(help="Model name to use.")
    ] = "gemini-2.0-flash",
    prompt_file: Annotated[
        str, typer.Option(help="Path to the prompt file.")
    ] = "PROMPT.md",
    schema_file: Annotated[
        str, typer.Option(help="Path to the schema file.")
    ] = "schema.json",
    batch_size: Annotated[
        int, typer.Option(help="Number of images to process in a batch.")
    ] = 2,
):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Google GenAI Client: {e}")
        return None

    print("--- Starting PlantCLEF Batch Analysis ---")

    # 1. Load Prompt Template
    print(f"Loading prompt from: {prompt_file}")
    prompt = load_text_file(prompt_file)

    # 3. Load Image Files
    print("Loading image files...")
    image_files = load_image_files()
    print(f"Loaded {len(image_files)} image files.")

    # 4. Create Batches
    image_batches = load_image_batches(image_files, batch_size=batch_size)
    print(f"Created {len(image_batches)} batches of images.")

    # 4. Run Generation
    generated_json = generate(
        client=client,
        image_paths=image_batches[0],
        prompt_template=prompt,
        model_name=model_name,
    )

    # 5. (Optional) Post-processing / Validation
    if generated_json:
        print("\n--- Final Generated JSON Output ---")
        # The output is already printed chunk by chunk, but we print the final accumulated one again
        # for clarity, or you could parse/validate it here.
        print("Final JSON output:")
        print(generated_json)


if __name__ == "__main__":
    app()
