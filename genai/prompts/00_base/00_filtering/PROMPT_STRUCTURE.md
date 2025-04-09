Objective: Parse the provided text, which contains entries for multiple plant species separated by '---'. For EACH species entry identified, generate a corresponding JSON object according to the item schema. Output a single JSON array containing all these generated objects, strictly conforming to the provided top-level array JSON schema.


JSON Schema:

```json
{
  "description": "List of structured outputs for initial species filtering, using the 'nullable' keyword (JSON Schema Draft 2020-12 style).",
  "type": "array",
  "items": {
    "description": "Schema for a single species entry.",
    "type": "object",
    "properties": {
      "species_id": {
        "description": "Unique identifier for the species, from the input data.",
        "type": "integer"
      },
      "scientific_name": {
        "description": "Scientific name of the species.",
        "type": "string"
      },
      "filter_is_vascular": {
        "description": "Is the plant determined to be vascular? (True=Yes, False=No, Null=Uncertain/NotFound).",
        "type": "boolean",
        "nullable": true
      },
      "filter_is_quadrat_relevant": {
        "description": "Is the plant determined to be relevant for a quadrat based on habit/size? (True=Yes, False=No, Null=Uncertain/NotFound).",
        "type": "boolean",
        "nullable": true
      },
      "filter_is_in_europe": {
        "description": "Is the plant determined to be present in Europe? (True=Yes, False=No, Null=Uncertain/NotFound).",
        "type": "boolean",
        "nullable": true
      },
      "filter_is_in_pyrenees_med": {
        "description": "Is the plant determined to be present in the Pyrenees or Med. Basin? (True=Yes, False=No, Null=Uncertain/NotFound).",
        "type": "boolean",
        "nullable": true
      },
      "justification": {
        "description": "Brief justification or notes supporting the filter determinations.",
        "type": "string",
        "nullable": true
      }
    },
    "required": [
      "species_id",
      "scientific_name",
      "filter_is_vascular",
      "filter_is_quadrat_relevant",
      "filter_is_in_europe",
      "filter_is_in_pyrenees_med"
    ]
  }
}
```