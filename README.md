# AI Monopoly JAX

A vectorized JAX-based Monopoly simulation and co-evolutionary training engine running on GPU/CPU.

---

## Environment Setup Instructions

This guide provides steps to set up the environment both **with Miniconda** and **without Miniconda (using standard Python venv)**.

### Option A: Setup with Miniconda (Recommended)

Miniconda is recommended as it helps manage separate environments and python versions cleanly.

1. **Create a new Conda Environment:**
   Create a dedicated environment for the project with Python 3.11:
   ```bash
   conda create -n monopoly python=3.11 -y
   ```

2. **Activate the Environment:**
   ```bash
   conda activate monopoly
   ```

3. **Install JAX (CUDA/GPU or CPU):**
   * **For GPU Support (CUDA 13):**
     If you have a CUDA 13 compatible GPU and drivers installed:
     ```bash
     pip install -U "jax[cuda13]"
     ```
     *(Or if using local CUDA Toolkit install: `pip install -U "jax[cuda13_local]"`)*
   
   * **For CPU Only:**
     If you do not have an NVIDIA GPU:
     ```bash
     pip install -U "jax[cpu]"
     ```

4. **Install Remaining Dependencies:**
   ```bash
   pip install streamlit plotly pandas matplotlib jinja2
   ```

---

### Option B: Setup without Miniconda (Using standard Python `venv`)

If you prefer not to use conda, you can use Python's built-in virtual environment (`venv`) manager. Make sure you have **Python 3.10 or 3.11** installed.

1. **Create a Virtual Environment:**
   Run the following in the project root directory:
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the Virtual Environment:**
   * **Linux/macOS:**
     ```bash
     source .venv/bin/activate
     ```
   * **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **Windows (CMD):**
     ```cmd
     .venv\Scripts\activate.bat
     ```

3. **Upgrade pip & Setup JAX (CUDA/GPU or CPU):**
   ```bash
   pip install --upgrade pip
   ```
   * **For GPU Support (CUDA 13):**
     ```bash
     pip install -U "jax[cuda13]"
     ```
   * **For CPU Only:**
     ```bash
     pip install -U "jax[cpu]"
     ```

4. **Install Other Dependencies:**
   Install using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

1. **Run Training:**
   ```bash
   python -m src.training.jax_train --run
   ```

2. **Start Dashboard:**
   ```bash
   streamlit run src/dashboard/app.py
   ```