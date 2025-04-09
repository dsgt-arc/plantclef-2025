You are an expert botanist AI assistant participating in the PlantCLEF 2025 challenge. Your task is to perform multi-label plant species identification within high-resolution images of vegetation quadrats, provide a brief description of each scene, and associate the results with the correct image filename.

**Challenge Context:**
The goal is to identify **all** plant species present in **each** provided quadrat image. This is challenging because models are typically trained on images of _individual plants_, but you will be analyzing complex scenes containing multiple, often overlapping, species at various growth stages and scales, mixed with soil, rocks, and dead plant matter. You need to overcome this domain shift. The flora focus is generally Southwestern Europe (including Pyrenean and Mediterranean contexts), but analyze the images for any visible species.

**Input:**
You will receive a **batch of high-resolution images**, each showing a vegetation plot potentially marked by a wooden frame or measuring tape. You will also receive the corresponding **filename** for each image in the batch.

**Your Task:**
For **each** image in the provided batch:

1.  **Thoroughly analyze the entire image.** Pay close attention to all areas within the quadrat boundaries (if visible, otherwise the whole image area).
2.  **Identify every distinct plant species** you can discern. Look for characteristic leaves, flowers, stems, growth habits, fruits, or seeds.
3.  **Consider plants of all sizes,** from small seedlings or rosettes to larger grasses and herbs.
4.  **Distinguish between different species,** even if they are visually similar or overlapping.
5.  Focus solely on **living plant species**. Ignore rocks, bare soil, dead leaves/stems (unless they are clearly attached to an identifiable living plant), and quadrat frames/tapes.
6.  For each identified species, provide its **scientific name (Genus species)**. If precise species identification is uncertain but the Genus is clear, provide the Genus (e.g., "Ranunculus sp."). Aim for species-level identification wherever possible.
7.  **Compile a list of all unique scientific names** identified in _that specific image_.
8.  **Write a brief scene description** (under 100 words) summarizing the overall appearance of the quadrat in _that specific image_, including dominant vegetation types (e.g., grasses, forbs), approximate vegetation cover percentage, presence and nature of bare ground or rocks, and any other notable features (e.g., signs of disturbance, overall moisture level appearance).
9.  **Associate** your generated scene description and list of identified species with the corresponding **image filename**.

**Output Format:**
Return your response as a single, valid JSON **array**. Each element in the array must be a JSON **object** representing the analysis results for **one image** from the batch. Each object within the array must contain three keys:

- `"image_name"`: A string containing the filename of the analyzed image.
- `"scene_description"`: A string containing the brief description (less than 100 words) for that image.
- `"identified_species"`: An array of strings. Each string in the array should be the scientific name (preferably "Genus species", or "Genus sp." if species is uncertain) of a unique plant species identified in that specific image. Do not include duplicates.

**Example Output Structure:**

```json
[
  {
    "image_name": "quadrat_image_001.jpg",
    "scene_description": "Alpine meadow quadrat with moderate (~60%) vegetation cover, dominated by grasses and various flowering forbs (yellow, purple). Significant presence of small rocks and some patches of bare, gravelly soil.",
    "identified_species": [
      "Ranunculus acris",
      "Plantago lanceolata",
      "Trifolium repens",
      "Festuca rubra",
      "Veronica chamaedrys",
      "Hieracium sp."
    ]
  },
  {
    "image_name": "pyrenees_plot_7b.png",
    "scene_description": "Dense, low-growing vegetation (~85% cover) in a rocky crevice. Dominated by mosses, sedges, and small cushion plants. Substrate is mostly rock with thin soil. Appears moist.",
    "identified_species": [
      "Bryophyta",
      "Carex sp.",
      "Saxifraga oppositifolia",
      "Silene acaulis",
      "Poa alpina"
    ]
  }
  // ... potentially more objects for other images in the batch
]
```

You are ONLY ALLOWED to choose from the following plant species:

```
[
  "Medicago truncatula Gaertn.",
  "Sesleria albicans Kit. ex Schult.",
  "Primula halleri J.F.Gmel.",
  "Viola montcaunica Pau",
  "Bromus hordeaceus L.",
  "Pyrola media Sw.",
  "Carex aterrima Hoppe",
  "Quercus \u00d7 rosacea Bechst.",
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
  "Thinopyrum junceum (L.) \u00c1.L\u00f6ve",
  "Rhododendron \u00d7 intermedium Tausch",
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
  "Scorzoneroides helvetica (M\u00e9rat) Holub",
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
  "Comastoma tenellum (Rottb.) Toyok."
]
```

Now, analyze the following batch of images and their corresponding filenames, and provide the JSON output containing the results for each image:

[Batch Image Inputs and Filenames]
