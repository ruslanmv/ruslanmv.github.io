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

#### For Windows Users 🪟

The best way to get a Linux-like experience on Windows is with **Git Bash**.

1.  **Install Git for Windows**: Download and install it from [git-scm.com](https://git-scm.com/download/win). This provides the **Git Bash** terminal, which you should use for all commands.
2.  **Install Python 3.11**: Download from the [official Python website](https://www.python.org/downloads/windows/). **Crucially, check the box that says "Add Python to PATH"** during installation.
3.  **Install GNU Make**: Open **PowerShell as an Administrator** and use a package manager like Scoop (recommended) or Chocolatey.
      * **With Scoop**:
        ```powershell
        Set-ExecutionPolicy RemoteSigned -Scope CurrentUser # Run this first if you get an error
        irm get.scoop.sh | iex
        scoop install make
        ```
      * **Or with Chocolatey**:
        ```powershell
        Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        choco install make
        ```

#### Clone the Project

With the prerequisites ready, open your terminal (**Git Bash** on Windows) and clone the project repository:

```bash
git clone https://github.com/ruslanmv/simple-environment.git
cd simple-environment
```

You now have all the configuration files you need to get started.

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

### The Future is Fast: An Optional Boost with `uv` ⚡️

For those who crave speed, there's a new tool called `uv`. It's an extremely fast Python package installer and resolver, written in Rust, that can serve as a drop-in replacement for `pip`.

  * **Why use it?** Speed. `uv` can be 10-100x faster than `pip`, making environment creation and updates nearly instantaneous.

#### Using `uv` with this Project

1.  **Install `uv`**:
      * **macOS/Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
      * **Windows**: `irm https://astral.sh/uv/install.ps1 | iex`
2.  **Create the environment with `uv`**:
    ```bash
    # Create the virtual environment
    uv venv
    # Activate it (same commands as before)
    source .venv/bin/activate
    # Install dependencies using uv
    uv pip sync pyproject.toml
    ```

You can even add a new target to your `Makefile` for this:

```makefile
# In your Makefile
install-uv:
	uv venv
	. .venv/bin/activate && uv pip sync pyproject.toml
```

Now you can just run `make install-uv` for a blazing-fast setup.

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
