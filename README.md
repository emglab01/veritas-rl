## Veritas RL (LGC‑V2 + PRINT) — training + environments

This repo is a lightweight, practical wrapper around the [`train/verl/`](./train/verl) submodule (VERL RLHF/RL training library), plus:

- **`environments/`**: single-turn verifiable environments (currently `lgc-v2`, `trace`)
- **`data_processing/`**: dataset generation, evaluation, and conversion scripts
- **`train/`**: training tools, scripts, and VERL integration
- **`evaluate/`**: model evaluation tools (deploy model to Chutes, then evaluate on affine environments)

---

## Quickstart

### 1. Clone (with submodules, VERL v0.5.x)

```bash
git clone --recurse-submodules git@github.com:<Repo>
cd veritas-rl

# (Recommended) Make sure VERL submodule is on the v0.5.x release line
cd train/verl
git checkout v0.5.x
cd ../..
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
cd train/verl && git checkout v0.5.x && cd ../..
```

### 2. Set Up Python Environment

**Data processing and evaluation scripts run directly on your host machine** (not in Docker).

**Quick Setup with uv:**

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup
./setup_uv.sh

# Configure environment
cp env.example .env
# Edit .env with your API keys
```

See [docs/setup_uv.md](docs/setup_uv.md) for detailed instructions.

### 3. Docker Setup (For Training)

**Training runs inside Docker container**:

```bash
cd train
./docker_run_verl.sh
```

This starts a GPU-enabled container. Training commands run inside the container.

- start a GPU-enabled container with the appropriate VERL image
- mount the repo at `/workspace/veritas-rl`
- set working directory to `/workspace/veritas-rl`
- drop you into a shell inside the container

**Inside the Docker container**, run training commands (see training section below).

**Note:** Docker is only needed for training. Data processing and evaluation run on your host machine.

---

## Dataset Generation

Generate datasets with problems and answers from LGC-V2 or Trace environments. You can generate datasets for specific task types, add reasoning using OpenAI, and evaluate the generated datasets to verify correctness.

For detailed documentation and command examples, see [docs/dataset_generation.md](docs/dataset_generation.md).

---

## Convert to Parquet

Convert JSONL datasets to VERL-compatible parquet format for training. Supports both RL format (with ground truth) and SFT format (with response field). The converter automatically detects response fields and preserves metadata.

For detailed options, schema information, and command examples, see [docs/dataset_generation.md](docs/dataset_generation.md).

---

## Training

Training is handled by the VERL submodule with wrapper scripts. The project supports two training approaches:

### SFT (Supervised Fine-Tuning)

SFT training fine-tunes models on pre-generated datasets in parquet format. The training process:
- Uses pre-generated parquet datasets (`train/parquet/train.parquet` and `val.parquet`)
- Fine-tunes the Qwen3-4B-Instruct model on prompt-response pairs
- Requires ≥2× A100 GPUs (160GB+)
- Logs metrics to WandB and saves checkpoints automatically
- Training scripts handle model downloading, configuration, and checkpoint management

### RL Training (GRPO)

RL training uses reinforcement learning with the GRPO (Group Relative Policy Optimization) algorithm:
- Uses dynamic environment-based datasets that generate challenges on-the-fly during training
- Evaluates model responses using verifiable reward functions from the environments
- No pre-generated dataset needed - challenges are generated dynamically
- Requires ≥4× H200 GPUs for optimal performance
- Custom reward functions route to the correct environment (LGC-V2 or Trace) based on task type

### Training Workflow

1. **Prepare Data**: Generate and convert datasets to parquet format (for SFT) or configure dynamic dataset (for RL)
2. **Start Docker Container**: Run `./train/docker_run_verl.sh` to enter the training environment
3. **Run Training Script**: Execute the appropriate training script inside the container
4. **Monitor Progress**: View metrics in WandB dashboard and check console output
5. **Checkpoints**: Model checkpoints are saved automatically at configured intervals

All training runs inside the Docker container and requires proper environment variables (HF_TOKEN, WANDB_API_KEY) to be set.

For detailed training documentation, scripts, configuration options, and WandB setup, see [docs/training.md](docs/training.md).

---

## Model Evaluation

Evaluate trained models on affine environments. The workflow involves deploying your model to Chutes (model serving platform) and then running evaluation scripts against the deployed model. Supports multiple evaluation methods including task IDs from file, single task evaluation, and range-based evaluation.

For detailed evaluation documentation, deployment instructions, and command examples, see [docs/evaluate_model.md](docs/evaluate_model.md).

---

## Environments

- **LGC-V2**: Deterministic logic puzzle generation with verification
- **Trace**: "Predict stdout" task generation

For environment details, see [docs/environments.md](docs/environments.md).

---

## Documentation

Detailed documentation is available in the [`docs/`](docs/) folder:

- [Dataset Generation](docs/dataset_generation.md) - Generate, evaluate, and convert datasets
- [Training](docs/training.md) - SFT and RL training guides
- [Model Evaluation](docs/evaluate_model.md) - Evaluation workflows
- [Environments](docs/environments.md) - Environment implementations
- [Setup Guide](docs/setup_uv.md) - Detailed setup instructions

---

## Security

- Use environment variables for API keys and tokens
- Never hardcode secrets in scripts
- See `env.example` for required environment variables


