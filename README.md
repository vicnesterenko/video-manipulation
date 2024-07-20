# Terminal-Manipulation-Video-App

This repository contains logic from `https://github.com/riffusion/riffusion` and own logic with generating video-manipulation task in `start.py`

This guide will walk you through the steps to set up and use the code.

### Installation

You have 2 methods to install this repo
1. **Clone this repo and install all locally**:

2. **Run `setup.cmd` and add file to `start.py` in clone repo and run it**

For 1 method make those commands:
1. **Clone the Repository.**
   Open a terminal and run the following command to clone this repository:

   ```bash
   git clone https://github.com/vicnesterenko/video-manipulation.git
   ```

2. **Navigate to the Directory.** Change to the project directory:

   ```bash
   cd riffusion-custom
   ```
3. **Install virtual environment**

   **On Windows:**

   Navigate to the directory where you want to create your project and virtual environment:
   ```terminal
   cd path\to\your\project\directory
    ```
   Create a virtual environment with the name `your_venv_name`:
   ```terminal
   python -m venv your_venv_name
    ```
   Activate the virtual environment:
   ```terminal
   your_venv_name\Scripts\activate
   ```
    **On macOS and Linux:**
   ```terminal
   cd path/to/your/project/directory
    ```
   Create a virtual environment with the name `your_venv_name`
   ```terminal
   python3 -m venv your_venv_name
    ```
   Activate the virtual environment:
   ```terminal
   source your_venv_name/bin/activate
   ```
   When you're done working in the virtual environment, you can deactivate it:
    ```terminal
   deactivate
   ```
   
4. **Install Dependencies.** Install the required dependencies using pip:

   ```terminal
   pip install -r requirements-custom.txt
   ```
5. **Run app**
   ```terminal
   python start.py
   ```

