## Prompt: Extract Structured Quadrat Scene Data from YAML with Error Correction

### Role:

You are a botanist and data validation assistant tasked with transforming vegetation plot descriptions from YAML format into structured JSON records.

Each YAML block describes a **single 50x50cm vegetation quadrat image** from the PlantCLEF challenge, focused on the Pyrenees and Mediterranean Basin. Your job is to **convert each YAML block into a corresponding JSON object**, correcting errors, inconsistencies, or formatting issues where necessary. All fields should be preserved and transformed 1:1. Do **not remove or skip entries**.

---

CSV data with up to date mapping of plot locations from filenames. To use the CSV data, use the prefix of the image_filename to get ecoregion name. The inferred regions must be consistent with the following information:

```csv
Short Identifier,Long Identifier,General Location,Ecological Description
2024-CEV3,2024 CEV3 Campaign,Likely Mediterranean region,Specific ecological details are not provided; further information is required.
CBN-can,Conservatoire Botanique National – Canigou,"Canigou Massif, Eastern Pyrenees",Montane Mediterranean flora with diverse plant communities.
CBN-PdlC,Conservatoire Botanique National – Pla de la Calme,"Pla de la Calme, Eastern Pyrenees",Alpine and subalpine vegetation with high biodiversity.
CBN-Pla,Conservatoire Botanique National – Plaine,Mediterranean plains,Mediterranean lowland flora with diverse plant species.
CBN-Pyr,Conservatoire Botanique National – Pyrénées,Pyrenees,Mountainous regions with alpine and subalpine plant communities.
GUARDEN-AMB,GUARDEN – Ambient Monitoring,Various Mediterranean sites,Diverse habitats monitored for ambient environmental conditions.
GUARDEN-CBNMed,GUARDEN – Conservatoire Botanique National Méditerranéen,Mediterranean region,Mediterranean flora with emphasis on conservation and biodiversity.
LISAH-BOU,LISAH – Le Boulou,"Le Boulou, Southern France",Agricultural landscapes with Mediterranean influences.
LISAH-BVD,LISAH – Basse Vallée de l'Aude,"Lower Aude Valley, Southern France",River valley ecosystems with Mediterranean vegetation.
LISAH-JAS,LISAH – Jas,"Jas, Southern France",Mediterranean agricultural and natural ecosystems.
LISAH-PEC,LISAH – Pécharmant,"Pécharmant, Southern France",Vineyard landscapes with Mediterranean flora.
OPTMix,OPTMix Experiment,Southern France,Experimental plots with mixed-species plantations to study biodiversity effects.
RNNB,Réserve Naturelle Nationale de la Baie,"Baie region, Southern France",Coastal and wetland habitats with diverse plant communities.
```

---

### Instructions:

For each YAML block:

1. Process each YAML block provided in the input sequentially and return a single JSON list containing one object for each processed block.
2. Validate and **correct** errors using prior knowledge and the schema constraints.
3. Ensure the final JSON object matches the format specified below, including data types, value ranges, and allowed enums.
4. **Do not discard entries** for any reason. If a value is invalid (e.g., “tall” instead of “< 15cm”), replace it with a best estimate.
5. **Inferred regions** (`metadata_inferred_region`) must be consistent with the Pyrenees or Mediterranean Basin. If the region is something that is not nearby, like Canada, then you should go back to the filename to try to infer the correct region in the context of mainland Spain or France. Only use inferred regions that are consistent with the CSV data provided.
6. **fields may be updated for new information**. You will not need to change any of the visual descriptions, but you may modify the reasoning fields and summary fields for correctness for any new changes made. If any changes are made to any fields, update the transcription changes field to reflect changes.

---

### JSON Schema Constraints:

- `image_filename`: string
- `image_capture_date`: string (YYYY-MM-DD format)
- `derived_season`: one of `"Spring"`, `"Summer"`, `"Autumn"`, `"Winter"`
- `metadata_inferred_region`: string (corrected to be within the Mediterranean or Pyrenees region in mainland Spain or France)
- `metadata_inferred_ecoregion`: string (from CSV data)
- `estimated_region_visual`: string
- `habitat_predicted_types`: list of strings
- `habitat_reasoning`: string (multi-line OK)
- `habitat_substrate_appearance`: list of strings
- `cover_veg_pct`, `cover_ground_pct`, `cover_rock_pct`, `cover_litter_pct`: numbers between 0–100
- `cover_reasoning`: string
- `non_plant_elements`: list of strings
- `veg_dominant_forms`: list of strings
- `veg_dominant_forms_reasoning`: string
- `veg_height_cm`: string (e.g., `"< 15cm"`, `"> 30cm"`)
- `veg_height_reasoning`: string
- `veg_canopy_structure`: string
- `veg_canopy_reasoning`: string
- `veg_woodiness_observed`: boolean
- `visual_phenology_notes`: string
- `trait_leaf_types`, `trait_flower_colors`, `trait_flower_shapes`, `trait_other_features`: list of strings
- `diversity_estimated_richness`: number
- `diversity_richness_confidence`: `"Low"`, `"Medium"`, or `"High"`
- `diversity_richness_reasoning`: string
- `taxa_suggested_groups`: list of strings
- `taxa_groups_reasoning`: string
- `summary_overall`: string
- `transcription_changes`: string (if any changes were made to the fields)

---

### Output:

For each YAML block, produce a list of valid JSON object using the above schema and rules. Example output:

```json
[{
  "image_filename": "CBN-Pyr-Plot-20230621.jpg",
  "image_capture_date": "2023-06-21",
  "derived_season": "Summer",
  "metadata_inferred_region": "Pyrenees",
  "metadata_inferred_ecoregion": "Conservatoire Botanique National – Pyrénées",
  "estimated_region_visual": "Rocky alpine slope",
  "habitat_predicted_types": ["Alpine meadow", "Subalpine grassland"],
  "habitat_reasoning": "Fine-leaved grasses and cushion plants indicate high-elevation herbaceous habitat.",
  "habitat_substrate_appearance": ["Rock fragments", "Mineral soil"],
  "cover_veg_pct": 55,
  "cover_ground_pct": 10,
  "cover_rock_pct": 30,
  "cover_litter_pct": 5,
  "cover_reasoning": "Cover values estimated visually from quadrat area.",
  "non_plant_elements": ["Wooden frame"],
  "veg_dominant_forms": ["Grasses", "Forbs"],
  "veg_dominant_forms_reasoning": "Tufted graminoids dominate the canopy with scattered broad-leaved forbs.",
  "veg_height_cm": "< 15cm",
  "veg_height_reasoning": "Most vegetation falls below top edge of quadrat frame.",
  "veg_canopy_structure": "Open",
  "veg_canopy_reasoning": "Visible soil and rock patches among plant clusters.",
  "veg_woodiness_observed": false,
  "visual_phenology_notes": "Peak flowering in forbs; active growth phase.",
  "trait_leaf_types": ["Linear", "Lobed"],
  "trait_flower_colors": ["Yellow", "White"],
  "trait_flower_shapes": ["Daisy-like", "Tubular"],
  "trait_other_features": ["Mosses in soil crevices"],
  "diversity_estimated_richness": 7,
  "diversity_richness_confidence": "Medium",
  "diversity_richness_reasoning": "Several morphologically distinct plant forms observed.",
  "taxa_suggested_groups": ["Asteraceae", "Poaceae", "Brassicaceae"],
  "taxa_groups_reasoning": "Family-level groups inferred from floral and leaf traits.",
  "summary_overall": "This is a mid-summer quadrat on a Pyrenean alpine slope. It features patchy vegetation dominated by grasses and small forbs with rocky substrate. Floral diversity is moderate, and phenology suggests peak season.",
  "transcription_changes": "No changes needed."
},
...
]
```
