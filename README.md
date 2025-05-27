This repository provides a modularized Python library and a set of executable scripts for time series anomaly detection, building upon techniques

**Original GitLab Repository:** https://gitlab.com/basu1999/tsfm-anomaly-paper

## Table of Contents

1.  [Features](#features)
2.  [Project Structure](#project-structure)
3.  [Installation](#installation)
    *   [Prerequisites](#prerequisites)
    *   [Cloning the Repository](#cloning-the-repository)
    *   [Docker Setup (Recommended)](#docker-setup-recommended)
    *   [Environment Setup (Inside Docker)](#environment-setup-inside-docker)
    *   [Dataset Setup](#dataset-setup)
    *   [Fold ID Setup](#fold-id-setup)
4.  [Usage: Running Scripts](#usage-running-scripts)
    *   [General Notes on Running Scripts](#general-notes-on-running-scripts)
    *   [Statistical & Tree-Based Methods Evaluation](#statistical--tree-based-methods-evaluation)
        *   [Run IQR Evaluation](#run-iqr-evaluation)
        *   [Run Modified Z-score Evaluation](#run-modified-z-score-evaluation)
        *   [Run Isolation Forest Evaluation](#run-isolation-forest-evaluation)
        *   [Run Local Outlier Factor (LOF) Evaluation](#run-local-outlier-factor-lof-evaluation)
    *   [Deep Learning Model Hyperparameter Optimization](#deep-learning-model-hyperparameter-optimization)
        *   [Run VAE Hyperparameter Optimization](#run-vae-hyperparameter-optimization)
        *   [Run MOMENT Fine-tuning Hyperparameter Optimization](#run-moment-fine-tuning-hyperparameter-optimization)
    *   [Training/Fine-tuning Final Deep Learning Models](#trainingfine-tuning-final-deep-learning-models)
        *   [Train Final VAE Model](#train-final-vae-model)
        *   [Fine-tune Final MOMENT Model](#fine-tune-final-moment-model)
5.  [Using the `tsfm_ad_lib` Library Programmatically](#using-the-tsfm_ad_lib-library-programmatically) (To be added)
6.  [Running Example Notebooks](#running-example-notebooks) (To be added)
7.  [Running Tests](#running-tests) (To be added)
8.  [Contributing](#contributing)
9.  [License](#license)
10. [Contact](#contact)

## Features

*   Modular and reusable Python library (`tsfm_ad_lib`) for time series anomaly detection tasks.
*   Implementation of various anomaly detection algorithms:
    *   Variational Autoencoder (VAE)
    *   Fine-tuned MOMENT models
    *   Isolation Forest
    *   Local Outlier Factor (LOF)
    *   Interquartile Range (IQR)
    *   Modified Z-score
*   Scripts for end-to-end workflows:
    *   Hyperparameter optimization using Optuna for VAE and MOMENT models.
    *   Evaluation of statistical and tree-based methods with threshold/parameter tuning.
    *   Training final deep learning models with optimized hyperparameters.
*   Configurable data paths, model parameters, and logging.
*   Dockerized environment for reproducibility.
## Project Structure 
```bash

tsfm-anomaly-paper/
├── tsfm_ad_lib/                # Core Python library for anomaly detection
│   ├── __init__.py             # Package initializer
│   ├── config.py               # Default configurations for models and training
│   ├── data_loader.py          # Data loading utilities and TimeDataset class
│   ├── preprocessing.py        # Data preprocessing and feature engineering functions
│   ├── models/                 # Model definitions (VAE, MOMENT, Tree-based, etc.)
│   │   ├── vae.py
│   │   ├── moment.py
│   │   └── tree_based.py
│   ├── training.py             # Training loop functions
│   ├── evaluation.py           # Evaluation and scoring utilities
│   └── utils.py                # Miscellaneous utility functions
│
├── scripts/                    # Executable pipeline scripts
│   ├── run_iqr_eval.py
│   ├── run_mz_score_eval.py
│   ├── run_isolation_forest_eval.py
│   ├── run_lof_eval.py
│   ├── run_vae_hyperopt.py
│   ├── run_moment_finetune_hyperopt.py
│   ├── train_final_vae.py
│   └── finetune_final_moment.py
│
├── notebooks/                  # Example Jupyter notebooks for experimentation
├── tests/                      # (To be added) Unit and integration tests
├── DATASET/                    # Input dataset directory
│   └── train.csv
├── lead-val-ids/               # Predefined K-fold validation ID files
│   └── val_id_fold0.pkl, ...
├── results/                    # Output directory for logs, models, and evaluation results
├── .gitignore                  # Git ignore rules
├── README.md                   # Project overview and usage guide
├── requirements.txt            # Python dependencies
└── LICENSE                     # Project license
```
---

# Installation Guide

## Prerequisites

* Git
* Docker (**Recommended** for consistent environment)
* NVIDIA GPU and NVIDIA Container Toolkit (if using GPU for deep learning models)

---

## Cloning the Repository

```bash
git clone https://gitlab.com/basu1999/tsfm-anomaly-paper.git  # Or your new repo URL
cd tsfm-anomaly-paper
```

---

## Docker Setup (Recommended)

This project uses a Docker environment to ensure reproducibility and manage dependencies, especially for GPU support.

### Install Docker

#### Windows & macOS:

Download and install from [https://www.docker.com/get-started](https://www.docker.com/get-started)

#### Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER  # Add current user to docker group (logout/login required)
```

---

### Install NVIDIA Container Toolkit (For GPU Support)

Follow official instructions at:
👉 [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Example for Ubuntu:

```bash
distribution=$( . /etc/os-release; echo $ID$VERSION_ID ) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Pull Docker Image

```bash
docker pull gcr.io/kaggle-gpu-images/python:latest
```

(Alternatively, you can build from a custom Dockerfile based on `nvidia/cuda` or `pytorch/pytorch`.)

---

## Run the Docker Container

From the project root (`tsfm-anomaly-paper/`):

### For GPU:

```bash
docker run --gpus all -it --rm \
  --name tsfm_anomaly_env \
  -v "$(pwd)":/workspace \
  -w /workspace \
  gcr.io/kaggle-gpu-images/python:latest \
  /bin/bash
```

### For CPU:

```bash
docker run -it --rm \
  --name tsfm_anomaly_env \
  -v "$(pwd)":/workspace \
  -w /workspace \
  gcr.io/kaggle-gpu-images/python:latest \
  /bin/bash
```

**Flags Explanation:**

* `--gpus all`: Enables GPU access (omit for CPU).
* `-it`: Interactive terminal.
* `--rm`: Removes container after exit.
* `-v "$(pwd)":/workspace`: Mounts current directory inside container.
* `-w /workspace`: Sets container working directory.

---

## Environment Setup (Inside Docker)

Once inside the Docker container, run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure `requirements.txt` includes:

* `optuna`
* `momentfm` (if used directly)
* `scikit-learn`
* `pandas`
* `numpy`
* `torch`
* `joblib`, etc.

---

## Dataset Setup

Download the **LEAD Dataset**:

👉 [Kaggle Competition Link](https://www.kaggle.com/competitions/energy-anomaly-detection/data?select=train.csv)

Create dataset directory and move the file:

```bash
mkdir -p DATASET
mv /path/to/your/downloaded/train.csv DATASET/
```

Ensure structure:

```
tsfm-anomaly-paper/
├── DATASET/
│   └── train.csv
```

---

## Fold ID Setup (For VAE & MOMENT Hyperparameter Optimization)

Required for `run_vae_hyperopt.py`, `run_moment_finetune_hyperopt.py`.

### Expected Directory:

```
tsfm-anomaly-paper/lead-val-ids/
```

Files needed:

* `val_id_fold0.pkl`
* ...
* `val_id_fold4.pkl`

If not provided, you can generate them using a script like `scripts/generate_fold_ids.py` (based on `sklearn.model_selection.KFold` on unique `building_ids` from `train.csv`).

---

## Usage: Running Scripts

All scripts are inside the `scripts/` directory. Run them **from inside the Docker container** and **from the project root (`/workspace`)**.

### Notes:

* **PYTHONPATH**: Either set `export PYTHONPATH=$PYTHONPATH:/workspace` or rely on the header inside scripts.
* **Output**: Default output directories are subfolders inside `results/` (e.g., `results/iqr_eval/`). Use `--output_dir` to override.
* **Logging**: Console + `.log` file per run. Use `--log_level DEBUG` or `INFO` as needed.
* **Help**: Scripts support `--help` to list options:

Example:

```bash
python scripts/run_iqr_eval.py --help
```

* **Data Path**: Override with `--data_path` if your `train.csv` is not in `DATASET/train.csv`.
* **Run scripts from** the project root: `/workspace`
* **All scripts live in**: `scripts/`
* **Set PYTHONPATH** (inside Docker):

  ```bash
  export PYTHONPATH=/workspace
  ```
* **Control logging with**:

  ```bash
  --log_level {DEBUG, INFO, WARNING, ERROR}
  ```
* **GPU/CPU control** (for DL models):

  ```bash
  --device cuda  # or cpu
  ```
---

## A. Statistical & Tree-Based Methods

These methods evaluate models **per building** to maximize F1 score. Results saved in `results/`.

### 1. Interquartile Range (IQR)

```bash
python scripts/run_iqr_eval.py \
  --output_dir results/iqr_evaluation \
  --k_search_values "0.5,4.5,0.5" \
  --log_level INFO
```

### 2. Modified Z-Score

```bash
python scripts/run_mz_score_eval.py \
  --output_dir results/mz_score_evaluation \
  --k_search_values "2.0,5.0,0.5" \
  --use_absolute_zscore \
  --log_level INFO
```

### 3. Isolation Forest

```bash
python scripts/run_isolation_forest_eval.py \
  --output_dir results/iforest_evaluation \
  --contamination_search_values "0.01,0.1,0.01" \
  --iforest_base_params '{"n_estimators": 100}' \
  --log_level INFO
```

### 4. Local Outlier Factor (LOF)

```bash
python scripts/run_lof_eval.py \
  --output_dir results/lof_evaluation \
  --contamination_search_values "0.01,0.1,0.01" \
  --lof_base_params '{"n_neighbors": 20}' \
  --log_level INFO
```

---

## B. Deep Learning: Hyperparameter Optimization

Requires `lead-val-ids/` directory with fold split `.pkl` files.

### 5. VAE Hyperparameter Search (with Optuna)

```bash
python scripts/run_vae_hyperopt.py \
  --output_dir results/vae_hyperopt_study \
  --study_name my_vae_study \
  --fold_id_dir lead-val-ids/ \
  --n_trials 10 \
  --epochs_min 5 --epochs_max 20 \
  --log_level INFO
```

### 6. MOMENT Hyperparameter Search

```bash
python scripts/run_moment_finetune_hyperopt.py \
  --output_dir results/moment_hyperopt_study \
  --study_name my_moment_study \
  --fold_id_dir lead-val-ids/ \
  --n_trials 5 \
  --epochs_min 5 --epochs_max 20 \
  --lr_min 1e-5 --lr_max 1e-2 \
  --log_level INFO
```

---

## C. Deep Learning: Final Model Training

Use best hyperparameters obtained above (`*_best_params.json`).

### 7. Train Final VAE

```bash
python scripts/train_final_vae.py \
  --hyperparams_path results/vae_hyperopt_study/my_vae_study_best_params.json \
  --output_dir results/final_models/vae \
  --model_filename trained_vae.pth \
  --log_level INFO
```

### 8. Fine-tune Final MOMENT

```bash
python scripts/finetune_final_moment.py \
  --hyperparams_path results/moment_hyperopt_study/my_moment_study_best_params.json \
  --output_dir results/final_models/moment \
  --model_savename finetuned_moment_pipeline \
  --log_level INFO
```

---

## Utilities

### Generate K-Folds (if `lead-val-ids/` is missing)

```bash
python scripts/generate_folds.py
```

## Future Work (Tests & Notebooks)
This is an initial modularized version of the codebase. Future enhancements will include:
*   **Comprehensive Automated Test Suite:** Unit Commands:** Double-check that the default paths and example commands align with your final file structure and how you intend users to run things and integration tests for the `tsfm_ad_lib` library using `pytest` to ensure robustness and correctness.
*   **Example Jupyter Notebooks:** Demonstrations of how to use the library components interactively for various tasks like data exploration. The paths in `lib_config.py` should be the ultimate source for defaults used by scripts.

---

## Contributing

Contributions are welcome! Please follow these steps:
1.  **`generate_folds.py`:** Decide if you want to include this script in the repository (e.g.,  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/YourNewFeature`).
3 in `scripts/` or root) and document its use in the "Fold ID Setup" section.
4..  Commit your changes (`git commit -m 'Add YourNewFeature'`).
4.  Push to the branch  **`requirements.txt` and `LICENCE` file:** Create these actual files in your project root.
5.  Open a Merge Request.

---

## License

Your choice*

---

## Contact

Name*
