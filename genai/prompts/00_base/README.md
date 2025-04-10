# base model

This is the base model, and the first model that we use.

The properties are the following:

- It does not use grounding.
- We have 8k species.
- We do the filtering task in a single shot using the cheapest model available.

## filtering

In our filtering task we have a two stage pipeline that works fairly well. We first generate a bunch of info, and then convert it into json.

```yaml
---
species_id: 1356200
scientific_name: Urtica dioica L.
vascular: Yes
quadrat_relevant: Yes
sw_europe_presence: Yes
pyrenees_med: Yes
justification: Common nettle, herbaceous, widespread in Europe including the Pyrenees and Mediterranean.

---
species_id: 1356400
scientific_name: Cosentinia vellea (Aiton) Tod.
vascular: Yes
quadrat_relevant: Yes
sw_europe_presence: Yes
pyrenees_med: Yes
justification: Fern species; Size and habit appropriate for quadrat sampling; Present in Southwest Europe, including Mediterranean areas.

---
species_id: 1356800
scientific_name: Arenaria hispanica Spreng.
vascular: Yes
quadrat_relevant: Yes
sw_europe_presence: Yes
pyrenees_med: Yes
justification: Herbaceous plant; Size and habit appropriate for quadrat sampling; Native to Spain.
```

Justification helps us debug what is going on. One field that was missing here was how common the species is, which would be useful for seeing what species are most likely to be found in the quadrats. We split this up across 200 requests that can be retried seveal times in parallel.

```json
[
  {
    "species_id": 1356200,
    "scientific_name": "Urtica dioica L.",
    "filter_is_vascular": true,
    "filter_is_quadrat_relevant": true,
    "filter_is_in_europe": true,
    "filter_is_in_pyrenees_med": true,
    "justification": "Common nettle, herbaceous, widespread in Europe including the Pyrenees and Mediterranean."
  },
  {
    "species_id": 1356400,
    "scientific_name": "Cosentinia vellea (Aiton) Tod.",
    "filter_is_vascular": true,
    "filter_is_quadrat_relevant": true,
    "filter_is_in_europe": true,
    "filter_is_in_pyrenees_med": true,
    "justification": "Fern species; Size and habit appropriate for quadrat sampling; Present in Southwest Europe, including Mediterranean areas."
  },
  {
    "species_id": 1356800,
    "scientific_name": "Arenaria hispanica Spreng.",
    "filter_is_vascular": true,
    "filter_is_quadrat_relevant": true,
    "filter_is_in_europe": true,
    "filter_is_in_pyrenees_med": true,
    "justification": "Herbaceous plant; Size and habit appropriate for quadrat sampling; Native to Spain."
  },
  ...
]
```

Getting a structure out of this is actually quite easy. What is disappointing is the output of this which shows that we can only filter down about 500 of the species.

```
species_id                    7806
scientific_name               7806
filter_is_vascular            7806
filter_is_quadrat_relevant    7510
filter_is_in_europe           7256
filter_is_in_pyrenees_med     6067
justification                 7806
dtype: int64
```

With a little bit of conditional logic:

```python
df_full = df.fillna(
    {
        "filter_is_quadrat_relevant": True,
        "filter_is_in_europe": True,
        "filter_is_in_pyrenees_med": True,
    }
)
# how many are not quadrat relevant or not in europe or not in pyrenees
cond = (
    ~df_full.filter_is_quadrat_relevant.astype(bool)
    | ~df_full.filter_is_in_europe.astype(bool)
    | ~df_full.filter_is_in_pyrenees_med.astype(bool)
)
df_full[cond].count()
```

This results in 445 entries. This means that the approach is not effective enough for us to do a light pass on the data and that we have to get more sophisticated information from each species.

## inference

On the inference side of things, we found that there are severe limits on the pro prompts that we can run at any given time. We tried doing inference 10 images at a time against all 8k species, but the actual output is poor since it keeps on repeating the same species over an over. However we have a 1500 rate limit per day on this, so we we can't exactly put one image at a time. This is probably untenable.

```json
[
  {
    "identified_species": [
      "Geranium molle L."
    ],
    "image_name": "CBN-PdlC-C6-20180905.jpg",
    "scene_description": "The quadrat is densely covered with low-growing vegetation, primarily characterized by small, rounded leaves. Some grass blades are scattered. The overall cover is high, with nearly complete vegetation. The presence of tape measures marking the plot is also visible."
  },
  {
    "identified_species": [],
    "image_name": "CBN-PdlC-C6-20190701.jpg",
    "scene_description": "The quadrat shows mostly bare ground covered in dead plant material. There are few small green plants. A rock is visible in the lower right corner of the frame. Measuring tapes delineate the plot boundaries."
  },
  {
    "identified_species": [
      "Taraxacum besarabicum (Hornem.) Hand.-Mazz.",
      "Geranium molle L."
    ],
    "image_name": "CBN-PdlC-C6-20190722.jpg",
    "scene_description": "This quadrat is covered with dense green vegetation, including grasses and small plants with rounded leaves. A dead plant stem extends across the center of the plot. A rock is present in the corner. There is also a dandelion seedhead."
  },
  ...
]
```

However what might work instead is describing each of the quadrats in detail. From this we can categorize what kinds of habitats we are looking at, and possibly reduce the number of species we actually have to consider.

## future work

There are a couple of directions that we'd like to take. First we'd like to do comprehensive scene understanding of all the images. This would let us localize the quadrat and better understand the habits. We can use a stronger reasoning model in order to achieve this, with quadrats grouped together. A second thing that would be useful is to get a better understanding of the possible species that we should look at based on the scene descriptions. We can use these descriptions to better filter out what we're looking (flowers of certain colors, short shrubs, etc.) and then use this to filter down the species list.

The reasoning models might be useful here since they use more context in order to whittle down on the appropriate answers. Additionally, it might be better to have some kind of RAG system instead of sticking everything into context (although its arguable on how well to quantify these things).
