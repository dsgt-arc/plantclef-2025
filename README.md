# plantclef-2025

![](./figures/plantclef-2025-banner.png)

## Paper | [arXiv](https://www.arxiv.org/abs/2507.06093)
We describe DS@GT's second-place solution to the [**PlantCLEF 2025 challenge**](https://www.kaggle.com/competitions/plantclef-2025/) on multi-species plant identification in vegetation quadrat images. 

Our pipeline combines:
1. A fine-tuned Vision Transformer **ViTD2PC24All** for patch-level inference.
2. A $\large 4 \times 4$ tiling strategy that aligns patch size with the network's $\large 518 \times 518$ receptive field.
3. Domain-prior adaptation through PaCMAP + K-Means visual clustering and geolocation filtering.

Tile predictions are aggregated by majority vote and re-weighted with cluster-specific Bayesian priors, yielding a macro-averaged F1 of 0.348 (private leaderboard) while requiring no additional training.


## Repository Structure

The repository is organized as follows:

```
root/
├── plantclef/        # Main codebase for the project
├── tests/            # Unit tests for the project
├── notebooks/        # Jupyter notebooks for data exploration and modeling
├── user/             # User-specific directories for experimentation
├── scripts/          # Utility scripts for data processing and automation
└── docs/             # Documentation for the project
```

### Key Directories

- **`plantclef/`**: Contains the core modules and submodules for the project.
- **`tests/`**: Includes test cases to ensure code quality and correctness.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis and prototyping.
- **`user/`**: A scratch space for users to experiment without affecting the main repository.
- **`scripts/`**: Scripts for tasks such as data preprocessing and model evaluation.
- **`docs/`**: Documentation explaining the code, datasets, and models.

---
