# mgustineli

> This is a scratch directory where you can commit files to without worrying about polluting the main repository.
> Use it to test out new ideas or to fork code/notebooks from other people and to share the results with them.

## 1. Project Setup
To organize your work on **PACE**, create a main `~/clef` folder. This will be your working directory for cloning repositories and managing project files.

### Steps to create the `clef` directory:
1. **Navigate to your home directory:**
    ```
    cd ~
    ```
    The output should look like: `/storage/home/hcoda1/9/user-name`
2. **Create the `clef` directory:**
    ```
    mkdir clef
    ```
3. **Navigate to the `clef` directory and clone your repository:** For example, to clone the `plantclef-2025` repository:
    ```
    cd clef
    git clone git@github.com:dsgt-kaggle-clef/plantclef-2025.git
    ```

**Note:** If you encounter an error during cloning, ensure your SSH key is added to your GitHub account (see the next section).

## 2. Authenticating to GitHub
To authenticate with GitHub via SSH, follow these steps to add your SSH key:

1. Display your public SSH key: 
    ```
    cat ~/.ssh/id_rsa.pub
    ```
    If the file doesn’t exist, generate an SSH key using:
    ```
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    ```
    This should be the email in your GitHub account.
2. Copy the output, which starts with `ssh-rsa` followed by a long string of characters.
3. Log in to [**GitHub**](https://github.com/). In the upper-right corner of your GitHub page, click your profile photo, then click **Settings**.
4. In the "Access" section of the sidebar, click **SSH and GPG keys**.
5. Click **New SSH key**. 
6. Give a key title, like `pace-ssh-key`.
7. In the "Key" field, paste your public key.
8. Click on **Add SSH Key**
9. Back on VS Code, set Git user information by running the commands:
    ```
    git config --global user.email "you@example.com"
    git config --global user.name "Your Name"
    ```
    Replace with your GitHub email and name.

If you're having issues, refer to the GitHub documentation [**Adding a new SSH key to your GitHub account**](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).


## 3. Setting Up a Virtual Environment
To keep dependencies isolated, create a virtual environment in your `~/scratch` directory and install the required packages.

1. Navigate to the scratch directory: `cd ~/scratch`
2. Create the Virtual Environment: `python -m venv .venv`
3. Activate the Virtual Environment: `source .venv/bin/activate`
4. Navigate to your repo: `cd ~/clef/plantclef-2025`
5. Intall the package in editable mode: `pip install -e .`
6. Install dependencies: `pip install -r requirements.txt`
7. Verify the installation: `pip list`
8. Run the package tests using the `pytest` command: `pytest -v tests/`
9. Add the pre-commit hooks to your repository to ensure that the code is formatted correctly and that the tests pass before committing: `pre-commit install`

Your environment is now set up and ready for development.
