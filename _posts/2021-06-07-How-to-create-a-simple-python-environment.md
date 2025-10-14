---
title: "How to create a simple Python Environment with Jupyter Notebook"
excerpt: "Master Your Workflow: The Definitive Guide to a Reproducible Python & Jupyter Environment"

header:
  image: "../assets/images/posts/2021-06-06-How-to-install-Jupyter-Notebook-in-Visual-Studio-Code/clement.jpg"
  teaser: "../assets/images/posts/2021-06-06-How-to-install-Jupyter-Notebook-in-Visual-Studio-Code/clement.jpg"
  caption: "Without big data analytics, companies are blind and deaf, wandering out onto the web like deer on a freeway.- Geoffrey Moore"
  
---

Hello everyone\! We've all been there: you start a new data science project, and half the battle is just getting your environment set up correctly. You wrestle with Python versions, conflicting packages, and that dreaded phrase: **"but it works on my machine\!"**

Today, we're going to put an end to that. This guide provides a deep dive into a simple yet incredibly robust project for creating a clean, reliable, and reproducible environment for Jupyter Notebook. We'll focus on a **local setup** as the primary method, empowering you to understand and control your environment directly. We'll also explore a **Docker-based approach** as a powerful alternative, perfect for ensuring absolute consistency across any machine.

This entire setup is available on [GitHub](https://github.com/ruslanmv/simple-environment). Let's dive in and build a flawless workflow. 🚀

### The Goal: A Clean Slate for Every Project

Our objective is to create an **isolated development environment**. Think of it as a pristine, dedicated workshop for each project. Nothing from your other projects (or your computer's main Python installation) can interfere with it. This practice is the cornerstone of professional software development, as it prevents dependency conflicts and makes your work portable and easy to share.

Before we start building, let's understand the tools that make this possible.

### Understanding the Core Concepts

#### What is a Development Environment?

At its core, an environment is the complete context in which your code runs. This includes the Python interpreter itself, all the installed libraries (like `pandas` or `matplotlib`), and system-level settings. A **Python virtual environment** creates an isolated folder containing a specific version of Python and its own set of libraries, separate from your system's global Python. This ensures that "Project A," which needs `pandas 1.5`, can coexist peacefully with "Project B," which requires `pandas 2.1`.

#### What is `pyproject.toml`? The Project Blueprint

`pyproject.toml` is the modern, standardized configuration file for Python projects. It's the single source of truth for your project's metadata and dependencies.

  * **How it Works**: The file is written in TOML (Tom's Obvious, Minimal Language), which is easy for humans to read. Inside, sections like `[project]` define your package's name and version, while the `dependencies` list specifies exactly which libraries are required. When a tool like `pip` or `uv` reads this file, it knows precisely what to install to make your project run. This replaces the older, less-structured `requirements.txt` file by providing a more robust and comprehensive standard.

#### What is a `Makefile`? The Automation Engine

A `Makefile` is essentially a recipe book for your command line. It allows you to define a series of complex shell commands and give them simple, memorable names (called "targets").

  * **How it Works**: You define a target, like `install:`, and then list the commands to be executed under it. When you type `make install` in your terminal, the `make` utility finds that target in the `Makefile` and runs the associated commands in order. This saves you from having to remember and type long, error-prone commands, streamlining your workflow and ensuring consistency.

-----

### Step 1: Laying the Foundation (Prerequisites & Setup)

Before we can build our environment, we need a few basic tools. Below are the specific commands to get everything you need on macOS, Linux, and Windows.

#### For macOS Users 

We'll use [Homebrew](https://brew.sh/), the de facto package manager for macOS. Open your terminal and follow these steps.

1.  **Install Homebrew** (if you don't have it):
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
2.  **Install Git, Python, and GNU Make**:
    ```bash
    brew install git python@3.11 make
    ```
      * **Important Note**: Homebrew installs GNU Make as `gmake`. To use the simpler `make` command, create an alias:
        ```bash
        echo 'alias make="gmake"' >> ~/.zshrc && source ~/.zshrc
        ```

#### For Linux Users (Debian/Ubuntu) 🐧

Use the `apt` package manager to get everything you need with a few commands.

```bash
sudo apt update
sudo apt install -y git python3.11 python3.11-venv make
```

### For Windows Users 🪟

The recommended setup is to use **Git Bash** for running commands and install the required tools using PowerShell. Using the correct terminal (**Regular** vs. **Administrator**) for each step is crucial.

-----

###  Step 1: Install Git and Python

You can install these quickly using `winget` in a standard PowerShell terminal.

1.  Open **Windows PowerShell** (not as Administrator).

2.  Run the following commands:

    ```powershell
    # Install Git (which includes the Git Bash terminal)
    winget install -e --id Git.Git

    # Install Python 3.11
    winget install -e --id Python.Python.3.11
    ```

    > **Note:** If you install Python manually from [python.org](https://www.python.org/downloads/windows/), it is essential that you check the box that says **“Add Python to PATH”** during installation.

-----

###  Step 2: Install GNU Make (Choose ONE Method)

Pick only one of the following package managers. The easiest method is using `winget`.

#### Method A: Winget (Easiest)

In a **REGULAR PowerShell**, run:

```powershell
winget install -e --id GnuWin32.Make
```

#### Method B: Scoop (User-Level Install)

This must be done in a **REGULAR PowerShell** (not Admin).

```powershell
# 1. Allow script execution for your user
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force

# 2. Install Scoop
irm get.scoop.sh | iex

# 3. Install Make
scoop install make
```

#### Method C: Chocolatey (System-Wide Install)

This must be done in an **ADMINISTRATOR PowerShell** 🛡️.

```powershell
# 1. Install Chocolatey (if you don't have it)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iwr https://community.chocolatey.org/install.ps1 -UseBasicParsing | iex

# 2. Install Make (in a new Admin terminal)
choco install make -y
```

-----

###  Step 3: Verify and Clone

1.  **Important:** Close all terminal windows and open a new **Git Bash** terminal to ensure all changes are applied.
2.  Verify the installations:
    ```bash
    git --version
    python --version
    make --version
    ```
3.  Clone the project repository and navigate into the directory:
    ```bash
    git clone https://github.com/ruslanmv/simple-environment.git
    cd simple-environment
    ```

You now have all the necessary tools and project files to get started.
-----

### The Main Path: Your Local Virtual Environment Workflow 🐍

This is the recommended starting point. It's fast, efficient, and teaches you the fundamentals of managing Python environments directly.

#### Step 2: Creating and Installing Dependencies

Now that we understand the key files, let's use our `Makefile` to bring our `pyproject.toml` blueprint to life. In your terminal, run:

```bash
make install
```

This single command tells `make` to:

1.  Check for Python 3.11.
2.  Create a `.venv` folder—your isolated virtual environment.
3.  Activate the environment and use `pip` to install the packages defined in `pyproject.toml`.

#### Step 3: Activating Your Environment

Your "workshop" is built, but you need to "step inside" to use it. This is called **activating the environment**.

  * **On macOS or Linux**:
    ```bash
    source .venv/bin/activate
    ```
  * **On Windows (using Git Bash)**:
    ```bash
    source .venv/Scripts/activate
    ```

Your terminal prompt will change to show `(.venv)`, confirming you're working inside the isolated environment.

#### Step 4: Launching Jupyter Notebook

With your environment active, start the Jupyter server:

```bash
jupyter notebook
```

This will launch the server and open the Jupyter dashboard in your browser. You're now running in a clean, reproducible local environment\!

-----

Of course. Here is the updated section with the additional explanation about `uv`'s capabilities.

-----

### The Future is Fast: An Optional Boost with `uv` ⚡️

For those who crave speed, there's a new tool called **`uv`**. It's an extremely fast Python package installer and project manager, written in Rust, that can replace `pip` and `venv`.

  * **Why use it?** **Speed**. `uv` can be **10-100x faster** than `pip` + `venv`, making environment creation and dependency installation nearly instantaneous.

#### How to Use `uv` with this Project

Since this project already has a `pyproject.toml` file, switching to `uv` is incredibly simple.

1.  **Install `uv`** (if you haven't already):

      * **macOS / Linux / Git Bash**:
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```
      * **Windows (PowerShell)**:
        ```powershell
        irm https://astral.sh/uv/install.ps1 | iex
        ```

2.  **Create the Environment & Install Dependencies**:
    The `uv sync` command does everything in one step: it creates the `.venv` virtual environment (if it doesn't exist) and installs all dependencies specified in `pyproject.toml`.

    ```bash
    # Run this single command in the project's root directory
    uv sync
    ```

3.  **Activate the Environment** (Optional):
    To use the environment in your interactive shell, activate it just like you would with a normal `venv`:

      * **macOS / Linux / Git Bash**:
        ```bash
        source .venv/bin/activate
        ```
      * **Windows (PowerShell)**:
        ```powershell
        .venv\Scripts\Activate.ps1
        ```

You can also add a simple, clean target to your `Makefile` for this:

```makefile
# In your Makefile, you could add this target:
uv-install: ## Create environment and install all dependencies with uv
	@echo "⚡️ Creating environment and syncing dependencies with uv..."
	@uv sync
	@echo "✅ Done! To activate, run: source .venv/bin/activate"
```
	Now you can just run `make install-uv` for a blazing-fast setup.
-----

#### A Glimpse of `uv`'s Full Power

Beyond just syncing an environment, `uv` is a complete, all-in-one toolchain that simplifies your entire Python workflow. Here are a few other common tasks you can manage with it:

  * **Add & Remove Packages**: `uv` can modify your `pyproject.toml` file for you, just like Poetry or PDM.

    ```bash
    # Add a new dependency
    uv add requests

    # Add a development-only dependency
    uv add --dev pytest

    # Remove a dependency
    uv remove requests
    ```

  * **Run Commands Without Activating**: `uv` can execute commands within the virtual environment without you needing to run `source .venv/bin/activate` first.

    ```bash
    # Run a Python script
    uv run python my_script.py

    # Run an installed tool like pytest or ruff
    uv run pytest
    uv run ruff check .
    ```

  * **Manage Python Versions**: It even has a built-in Python version manager, replacing the need for tools like `pyenv`.

    ```bash
    # Install a specific Python version
    uv python install 3.12

    # List all installed versions
    uv python list
    ```

In short, `uv` consolidates many different tools (`pip`, `venv`, `pip-tools`, `pyenv`) into a single, lightning-fast binary.

-----

### The Alternative Path: The Docker Workflow (Your Escape Hatch) 🐳

What if a local setup isn't enough? For maximum reproducibility and zero host-system interference, we turn to **Docker**.

#### What is a Container, and How Does it Work?

A **container** is a lightweight, standalone, executable package of software that includes everything needed to run it: code, runtime, system tools, libraries, and settings.

  * **The Shipping Container Analogy**: Think of a physical shipping container. It doesn't matter what's inside (electronics, food, etc.) or where it's going (ship, train, truck)—the container itself is standardized. Software containers do the same for code. They package an application and its dependencies into a standardized unit that can run consistently on any machine that has Docker installed.
  * **Containers vs. Virtual Machines (VMs)**: A VM virtualizes an entire hardware stack, including a full guest operating system, making it heavy and slow to start. A container, on the other hand, virtualizes the operating system itself, sharing the host machine's kernel. This makes containers extremely lightweight, fast, and efficient.

The workflow involves three key parts:

1.  **`Dockerfile`**: The recipe for building our environment.
2.  **Image**: A read-only template created from the `Dockerfile`. `make build-container` creates this.
3.  **Container**: A runnable instance of an image. This is the "live" environment created by `make run-container`.

#### Step 5: Building the Docker Image

This command reads the `Dockerfile` and builds the self-contained image.

```bash
make build-container
```

#### Step 6: Running the Jupyter Container

This command launches a live container from your image.

```bash
make run-container
```

This powerful command starts the container, maps port `8888` for browser access, and **mounts** your local project folder into the container. This crucial step means your local files and the container's files are synchronized—any notebook you save in Jupyter is instantly saved on your computer.

Open your browser and navigate to **http://localhost:8888** to access your fully containerized, perfectly reproducible Jupyter environment.

By mastering this local-first approach and keeping Docker as your ace in the sleeve, you can build a development process that is clean, efficient, and truly reproducible.

**Congratulations!** We have learned how to install a simple Python environment.
