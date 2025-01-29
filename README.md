# plantclef-2025

This is a template repository for CLEF projects. It is meant to be forked and used as a starting point for new projects.

## TODOS
Refer to the [**`Project-Setup`**](https://github.com/dsgt-kaggle-clef/dsgt-clef-docs/blob/main/Project-Setup-2025.md) guide.


## Structure

The repository structure is as follows:

```
root/
├── plantclef/        # the task package for the project
├── tests/            # tests for the project
├── notebooks/        # notebooks for the project
├── user/             # user-specific directories
├── scripts/          # scripts for the project
└── docs/             # documentation for the project
```

The `plantclef` directory should contain the main code for the project, organized into modules and submodules as needed.
The `tests` directory should contain the tests for the project, organized into test modules that correspond to the code modules.
The `notebooks` directory should contain Jupyter notebooks that capture exploration of the datasets and modeling.
The `user` directory is a scratch directory where users can commit files without worrying about polluting the main repository.
The `scripts` directory should contain scripts that are used in the project, such as utility scripts for working with PACE or GCP.
The `docs` directory should contain documentation for the project, including explanations of the code, the data, and the models.
