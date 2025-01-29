# mgustineli

> This is a scratch directory where you can commit files to without worrying about polluting the main repository.
> Use it to test out new ideas or to fork code/notebooks from other people and to share the results with them.

## 1. Project Setup
To organize your work on **PACE**, create a main `~/clef` folder. This will be your working directory for cloning repositories and managing project files.

### Steps to create the `clef` directory:
1. **Navigate to your home directory:** `cd ~`

    The output should look like: `/storage/home/hcoda1/9/user-name`
2. **Create the `clef` directory:** `mkdir clef`
3. **Navigate to the `clef` directory and clone your repository:** For example, to clone the `plantclef-2025` repository:
    ```
    cd clef
    git clone git@github.com:dsgt-kaggle-clef/plantclef-2025.git
    ```

**Note:** If you encounter an error during cloning, ensure your SSH key is added to your GitHub account (see the next section).


## 2. Downloading the GitHub CLI `gh`
To authenticate to GitHub, first we will download and install the GitHub CLI (`gh`) 
directly using `curl` and add it to PATH, follow these steps:

1. Create a `bin` directory in your home folder:
    ```
    mkdir -p ~/bin
    ```
2. Get exact download URL via GitHub API:
    ```
    LATEST_URL=$(curl -s https://api.github.com/repos/cli/cli/releases/latest | grep "browser_download_url.*linux_amd64.tar.gz" | cut -d '"' -f 4)
    ```
2. Download the `gh` binary file into the `~/bin` directory using curl:
    ```
    curl -L "$LATEST_URL" -o ~/bin/gh.tar.gz
    ```
3. Extract the file using `tar`:
    ```
    tar -xzf ~/bin/gh.tar.gz -C ~/bin --strip-components=2 gh_*/bin/gh
    ```
4. Remove the downloaded `tar.gz` file:
    ```
    rm ~/bin/gh.tar.gz
    ```
5. Add the `~/bin` directory to your PATH by editing your shell configuration file. Add the following line at the end of the file:
    ```
    export PATH="$HOME/bin:$PATH"
    ```
6. Apply the changes to your current session:
    ```
    source ~/.bashrc
    ```
Now you should be albe to run `gh` from any directory. To verify the installation, run: 
```
gh --version
```

## 3. Authenticating to GitHub using GitHub CLI
This section streamlines the authentication process to GitHub using the 
GitHub CLI (`gh`), which simplifies the SSH setup. 

### Step 1: Authenticate to GitHub using GitHub CLI
**Run in PACE (VS Code terminal)**

1. **Start the Authentication Process:** Run `gh auth login` to begin the authentication process.
2. **Choose SSH for Git Operations:** When prompted, select **SSH** as the preferred protocol for Git operations.
3. **Generate a New SSH Key (if required):** If you don't already have an SSH key, `gh` will prompt you to generate one. Follow the on-screen instructions to create a new SSH key.
4. **Authenticate Your SSH Key with GitHub:** `gh` will automatically add your SSH key to your GitHub account. Follow any additional prompts to complete the process.
5. **Verify Authentication:** After completing the setup, run `gh auth status` to check if you're successfully authenticated.

### Step 2: Verify GitHub User Information (Optional)

**Run in PACE (VS Code terminal)**

It's good practice to ensure your Git identity is correctly set:
1. **Check Git Configurations:** Run `git config --list` to see your Git configurations, including user name and email.
2. **Set Git User Information If Not Set:** If not already set, configure your Git user information:
```
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```
Replace with your GitHub email and name.


## 4. Cloning the Repository and Branching for Development
This final step in the onboarding process involves cloning the desired GitHub repository to your VM and creating a new branch for development. In this example, we'll use the [**`plantclef-2025`**](https://github.com/dsgt-kaggle-clef/plantclef-2025/tree/main) repository.

### Step 1: Clone the Repository
1. **Navigate to Desired Directory:** Choose the directory where you want to clone the repository. For instance, `cd ~/clef`.
2. **Clone the Repository:** Run the following command to clone the `plantclef-2025` repository:
    ```
    git clone git@github.com:dsgt-kaggle-clef/plantclef-2025.git
    ```
3. **Navigate into the Repository Directory:** After cloning, move into the repository's directory:
    ```
    cd plantclef-2025
    ```

### Step 2: Create and Checkout a New Branch
Creating a new branch ensures that your development work is separated from the main branch, allowing for easier code management and review.

1. **Fetch All Branches (Optional):** If you want to see all existing branches first, run `git fetch --all`.
2. **Create a New Branch:** Create a new branch off the main branch for your development work:
    ```
    git checkout -b username/your-branch-name
    ```
    Replace `your-branch-name` with a meaningful name for your development work, typically like `username/data-analysis`, or similar prefixes.
3. **Verify New Branch:** Ensure you're on the new branch with `git branch`. The new branch should be highlighted.


### Step 3: Verify the Setup
1. **Check Repository Content:** Verify that the repository content is correctly cloned by listing the files with `ls`.
2. **Check Branch Status:** Use `git status` to ensure you're on the correct branch and to see if there are any changes.


### Step 4: Naming Convention for Jupyter Notebooks
When creating new Jupyter notebooks, adhere to the following naming convention:

- **Format:** `initials-date-version-title.ipynb`
- **Explanation:**
    - **initials:** Your initials (e.g., Tony Stark → ts).
    - **date:** The date you created the notebook in YYYYMMDD format.
    - **version:** A two-digit version number, starting from `00`.
    - **title:** A brief, hyphen-separated title describing the notebook's purpose.

**Example**
If Tony Stark creates a data analysis notebook on January 18, 2025, 
the filename would be:
- `ts-20250118-00-data-analysis.ipynb`

### Step 5: Regularly Commit Changes
Remember to regularly commit your changes to maintain a record of 
your work and to synchronize with the remote repository.

1. **Stage Changes:** Use `git add .` to stage all changes in the 'notebooks' directory.
2. **Commit Changes:** Commit with a descriptive message:
    ```
    git commit -m "Add initial data analysis notebook by TS"
    ```
3. **Push to Remote:** Push your changes to the remote repository:
    ```
    git push -u origin username/your-branch-name
    ```


## 5. Setting Up a Virtual Environment
To keep dependencies isolated, we will create a virtual environment in your `~/scratch` directory and install the required packages.

### Prerequisite: Make sure you have done the TODOs in the [**`clef-project-template`**](https://github.com/dsgt-kaggle-clef/plantclef-2025/tree/main) repo!

After finishing the TODOs from the `clef-project-template`, do the following: 
1. Navigate to the scratch directory: `cd ~/scratch`
2. Create the Virtual Environment: `python -m venv .venv`
3. Activate the Virtual Environment: `source .venv/bin/activate`
4. Navigate to your repo: `cd ~/clef/plantclef-2025`
5. Intall the package in editable mode: `pip install -e .`
6. Install dependencies: `pip install -r requirements.txt`
7. Verify the installation: `pip list`
8. Run the package tests using the `pytest` command: `pytest -v tests/`
9. Add the **pre-commit hooks** to your repo. This ensures that the code is formatted correctly and that the tests pass before committing: `pre-commit install`

Your environment is now set up and ready for development.


That's it! You're now all set to start developing on your project. Happy coding! 😊💻





<!--1. **Navigate to the scratch directory:** 
    ```
    cd ~/scratch
    ```
2. **Create the Virtual Environment:**
    ```
    python -m venv .venv
    ```
3. **Activate the Virtual Environment:** 
    ```
    source .venv/bin/activate
    ```
4. **Navigate to your repo:** 
    ```
    cd ~/clef/plantclef-2025
    ```
5. **Intall the package in editable mode:**
    ```
    pip install -e .
    ```
6. **Install dependencies:** 
    ```
    pip install -r requirements.txt
    ```
7. **Verify the installation:**
    ```
    pip list
    ```
8. **Run the package tests using the `pytest` command:** 
    ```
    pytest -v tests/
    ```
9. **Add the pre-commit hooks to your repo:** This ensures that the code is formatted correctly and that the tests pass before committing. 
    ```
    pre-commit install
    ``` -->

