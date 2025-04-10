Objective: For each plant species listed below, determine if it meets the core filtering criteria for the PlantCLEF 2025 Pyrenean/Mediterranean quadrat challenge. Provide simple Yes/No/Uncertain answers based on grounded search. You are given rows in the form of CSV wit species_id,species,genus,family.

Required Determinations per Species:
a. Is Vascular Plant? (Yes/No/Uncertain based on classification like Angiosperm, Fern, etc.)
b. Is Quadrat Relevant? (Yes/No/Uncertain based on typical habit/size - e.g., herb, grass, small shrub, seedling, generally NOT large mature trees unless seedling form common in quadrats)
c. Is Present in Southwestern Europe? (Yes/No/Uncertain based on native/naturalized range)
d. Is Present in Pyrenees/Mediterranean? (Yes/No/Uncertain based on specific regional distribution)
e. Short justification of less than 60 words that best explains the determination.

Output Format Instructions:
- Use YAML format for the output that is not inside code blocks.
- Process each species sequentially.
- Under each species name, list the determinations clearly:
    * species_id: [species_id]
    * scientific_name: [species]
    * vascular: [Yes/No/Uncertain]
    * quadrat_relevant: [Yes/No/Uncertain]
    * sw_europe_presence: [Yes/No/Uncertain]
    * pyrenees_med: [Yes/No/Uncertain]
    * justification: [Short justification <60 words]
- Use `---` as a clear separator between species entries.
- Ensure the number of inputs matches the number of species provided.