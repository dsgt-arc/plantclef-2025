# mgustineli

> This is a scratch directory where you can commit files to without worrying about polluting the main repository.
> Use it to test out new ideas or to fork code/notebooks from other people and to share the results with them.

## Project Setup
### Directories & Repo
We will create a main `clef` folder where we will be working from. This is the place where we will clone the repos and work from.

To create the `clef` directory, make sure you are in your home directory. Run the commands `cd ~` followed by `pwd`.You should get something like `/storage/home/hcoda1/9/user-name`.

Create the `clef` directory by running: `mkdir clef`.

Navigate to the `clef` directory and clone your repo. I'll be cloning the `plantclef-2025` repo as an example:
```
cd clef
git clone git@github.com:dsgt-kaggle-clef/plantclef-2025.git
```

If you get an error when cloning the repo, it's likely because you don't have the `id_rsa.pub` in your GitHub account.

Follow the next steps to add a new SSH key to your GitHub account:
1. Run the command: `cat ~/.ssh/id_rsa.pub`
2. Copy the output. It should be a string starting with `ssh-rsa` followed by a long string of letter and numbers, like `ssh-rsa ABCDE123...`
3. Go to your GitHub account. Click on your profile on the top right corner. Select `Settings` and `SSH and GPG keys`
4. Click on `New SSH key`. Give a key title, like `pace-ssh-key`, and paste the output from the previous step in the Key box. Click on `Add SSH Key`.

### Virtual Environment
We will set up a virtual environment in your `scratch` directory and install the packages from your `pyproject.toml` file that's in the repo. Follow these steps:

**1. Navigate to the Scratch Directory:** Move into the directory where you want to create the virtual environment:
```
cd ~/scratch
```

**2. Create the Virtual Environment:** Create a virtual environment in the `scratch` directory:
```
python3 -m venv .venv
```
This will create a virtual environment named `.venv` in `~/scratch`.

**3. Activate the Virtual Environment:** Activate the virtual environment:
```
source ~/scratch/.venv/bin/activate
```

**4. Navigate to Your Repository:** Move into the plantclef-2025 directory:
```
cd ~/clef/plantclef-2025
```

Install Dependencies: Use pip to install the packages listed in your pyproject.toml:

pip install -r requirements.txt

If you're using Poetry or another tool that works with pyproject.toml, install the dependencies using the appropriate command (e.g., poetry install).

Verify Installation: Check that the packages were installed correctly:

pip list