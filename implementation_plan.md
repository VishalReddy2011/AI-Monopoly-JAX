# Implementation Plan: Vectorized JAX Monopoly on GPU

This plan outlines the architecture and execution steps to rewrite the Monopoly simulation and training engine in JAX. By vectorizing the state representation and game mechanics, we can run thousands of games in parallel natively on the GPU (inside WSL2 Ubuntu-22.04), bypassing CPU bottlenecks completely and accelerating co-evolutionary training by 100x+.

## User Review Required

> [!IMPORTANT]
> **Environment Change (WSL2)**:
> Since native Windows JAX does not support GPU/CUDA acceleration, we will set up and run the training pipeline inside your WSL2 `Ubuntu-22.04` environment using a new conda environment named `monopoly` with Python 3.11.
>
> **Algorithm Transition (NEAT -> Fixed MLP with ES)**:
> Since NEAT (evolving dynamic network topologies) breaks JAX's static-shape compile requirements, we will switch to a **fixed-topology Multi-Layer Perceptron (MLP)** trained using a JAX-native **Evolutionary Strategy (ES) / Genetic Algorithm**.
>
> **Model Shape**:
> - Inputs: 37 features (same as the existing personality-switched model).
> - Outputs: 7 action activations (Buy, Build, Mortgage, Unmortgage, Accept Trade, Trade Counter Premium, Bidding Value).
> - Hidden Layer: 1 layer of 64 units (or similar).

---

## Proposed Changes

### 1. Environment & Setup

#### [NEW] [setup_wsl_env.sh](file:///c:/Vishal/Projects/JAX%20Monopoly/setup_wsl_env.sh)
A bash script to automate environment creation inside WSL2:
* Create a conda env `monopoly` with `python=3.11`.
* Install CUDA-enabled JAX: `pip install -U "jax[cuda12]"` (compiling for your CUDA 13.x/12.x compatible driver).
* Install supporting packages: `pip install streamlit plotly pandas matplotlib jinja2`.

---

### 2. JAX-Vectorized Simulator

#### [NEW] [jax_game.py](file:///c:/Vishal/Projects/JAX%20Monopoly/src/simulator/jax_game.py)
Implement a pure, side-effect-free simulator in JAX:
* **`MonopolyState` PyTree**: Stores all game state variables as JAX arrays (cash, positions, jail status, cards, property owners, house counts, mortgage status, turn number, active player, random keys).
* **Static Board Arrays**: Define costs, house costs, rents, and color groups as constants.
* **`game_step(state, mlp_weights)`**:
  * Simulates active player rolling dice, moving, landing, paying rent, buying properties.
  * Resolves auctions in one step by checking maximum bids from players using their policies.
  * Resolves liquidity crises by selling houses and mortgaging properties in a JIT-compatible loop.
  * Resolves multi-round trade proposals by checking proposer & target policies in a deterministic sequence of evaluations.
* **`play_game_scan(rng_key, mlp_weights, max_turns=300)`**: Uses `jax.lax.scan` to run a full game of 300 turns at GPU speed without any CPU overhead.
* **`vmap` support**: Enables parallel execution of games across different seeds/environments.

---

### 3. Evolutionary Strategy & Co-Evolution Training

#### [NEW] [jax_train.py](file:///c:/Vishal/Projects/JAX%20Monopoly/src/training/jax_train.py)
* **MLP Forward Pass**: A pure static forward pass implementation using weights from a PyTree (or flat array of parameters).
* **Fitness Evaluation**: Runs parallel self-play games for each genome in the population. The 4 players use the same genome but receive different personality vectors (appended to inputs at indices 30-36). Calculates role-specific fitness (Industrialist, Jailbird, Scrooge, Flipper) and averages them.
* **Evolution Loop**:
  * Implement standard Genetic Algorithm mutation (adding Gaussian noise to weights), crossover, and elitism selection.
  * Evolve a population of 50-100 genomes for 200 generations.
  * Save the best agent's weights as `data/best_jax_agent.pkl`.
  * Save training progress metrics to `data/neat_stats.json` (or `data/jax_stats.json`).
  * Run a showcase game and output logs in the exact same format to `data/latest_game_log.json` to keep dashboard replay functional.

---

### 4. Dashboard & Dashboard Compatibility

#### [MODIFY] [app.py](file:///c:/Vishal/Projects/JAX%20Monopoly/src/dashboard/app.py)
* Update model loading to detect the JAX champion pickle (which is a dictionary of MLP weights).
* Implement a simple NumPy fallback for the MLP forward pass to evaluate the champion in real-time on the Streamlit client.
* Update the network visualizer to draw a beautiful Dense MLP neural graph (color-coded connection weights, clear input/output labels) instead of the NEAT graph.
* Keep the replay board and log viewer working since JAX-generated logs will match the JSON structure exactly.

---

## Verification Plan

### Automated Tests
* Run `setup_wsl_env.sh` to initialize the environment and run a quick verification command to test JAX GPU visibility inside WSL2:
  `wsl -d Ubuntu-22.04 -- bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate monopoly && python -c 'import jax; print(\"Devices:\", jax.devices())'"`
* Run a dry run of the JAX training script for 2 generations to confirm:
  1. The environment initializes without error.
  2. The game loop compiles and executes on the GPU.
  3. Fitness metrics and logs are written correctly.

### Manual Verification
* Run the Streamlit dashboard (`streamlit run src/dashboard/app.py` or through WSL) and verify:
  1. The interactive champion sliders correctly activate the JAX model outputs.
  2. The JAX-evolved latest game replay plays correctly on the HTML board grid.
