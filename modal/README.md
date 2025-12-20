# Modal Support for MTSA-RLVR

This directory contains scripts to run MTSA-RLVR training on [Modal](https://modal.com/).

## Structure

- `training.py`: Modal application defining the environment, mounts, and training functions.
- `requirements.txt`: Modal-specific requirements.

## Setup

1.  **Install Modal**:
    ```bash
    pip install modal
    ```

2.  **Authentication**:
    ```bash
    modal setup
    ```

## Usage

You can run the training remotely on Modal's A100 GPUs using the following commands:

### Attack Training
```bash
modal run modal/training.py --mode attack --model Qwen/Qwen2.5-7B-Instruct
```

### Defense Training
```bash
modal run modal/training.py --mode defence --model Qwen/Qwen2.5-7B-Instruct
```

## Configuration

- **GPU**: Defaults to a single A100. You can modify the `@app.function` decorator in `training.py` to change GPU type or count.
- **Volumes**:
  - `/models`: Persistent volume for saving checkpoints (named `mtsa-models`).
  - `/data`: Persistent volume for datasets (named `mtsa-datasets`).
- **Mounts**: The `MTSA/` directory is automatically mounted to `/workspace/MTSA` in the container.
