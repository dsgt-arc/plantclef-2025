This is the base model, and the first model that we use.

The properties are the following:

- It does not use grounding.
- We have 8k species.
- We do the filtering task in a single shot using the cheapest model available.

We found that there are severe limits on the pro prompts that we can run at any given time. We tried doing inference 10 images at a time against all 8k species, but the actual output is poor since it keeps on repeating the same species over an over. However we have a 1500 rate limit per day on this, so we we can't exactly put one image at a time. This is probably untenable.

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
