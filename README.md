# clef-project-template

This is a template repository for CLEF projects. It is meant to be forked and used as a starting point for new projects.

## TODOS

1. Fork this repository.
2. Update the `my_task_package` directory with the name of your task package e.g. `birdclef` or `longeval`.
3. Update the `pyproject.toml` file with the following elements:
   1. `project:name` should be a hyphenated version of the task package
   2. `project:authors` should be updated with the names of the team members
   3. `project:urls` should be updated with the corresponding URLs for the project
   4. `tools.setuptools.packages.find:include` should be updated with the directory name of the task package
4. Update `user/my-username` and the corresponding `README.md` file with the username and a description of the user's directory.

## quickstart

Install the package into your environment.
It is useful to install the package in editable mode so that you can make changes to the code and see the changes reflected in your environment.
Use a virtual environment when possible to avoid conflicts with other packages.

```bash
# create the virtual environment
python -m venv .venv

# activate the virtual environment
source .venv/bin/activate

# install the package in editable mode
pip install -e .
```

To run the package tests, use the `pytest` command.

```bash
pytest -v tests/
```

Add the pre-commit hooks to your repository to ensure that the code is formatted correctly and that the tests pass before committing.

```bash
pre-commit install
```

## structure

The repository structure is as follows:

```
root/
├── my_task_package/  # the task package for the project
├── tests/            # tests for the project
├── notebooks/        # notebooks for the project
├── user/             # user-specific directories
├── scripts/          # scripts for the project
└── docs/             # documentation for the project
```

The `my_task_package` directory should contain the main code for the project, organized into modules and submodules as needed.
The `tests` directory should contain the tests for the project, organized into test modules that correspond to the code modules.
The `notebooks` directory should contain Jupyter notebooks that capture exploration of the datasets and modeling.
The `user` directory is a scratch directory where users can commit files without worrying about polluting the main repository.
The `scripts` directory should contain scripts that are used in the project, such as utility scripts for working with PACE or GCP.
The `docs` directory should contain documentation for the project, including explanations of the code, the data, and the models.
