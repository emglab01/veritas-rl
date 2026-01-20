# Evaluation Scripts

This directory contains scripts for evaluating models on affine environments.

**Workflow**: Deploy model → Evaluate model
1. First, deploy your model to Chutes using tools in the `benchmark` folder
2. Then, run evaluations using `evaluate_local_model.py` in this folder

## Prerequisites

### 1. Submodule Setup

The `affine-cortex` submodule must be initialized and updated:

```bash
# From the project root
cd /root/veritas-rl
git submodule update --init --recursive
```

The `affine-cortex` submodule is located at:
- Path: `evaluate/affine-cortex`
- Repository: https://github.com/affinefoundation/affine-cotex.git

### 2. Virtual Environment Setup (uv)

**Important**: Both `affine-cortex` and `scripts` folders share the same virtual environment, which must be located in the `affine-cortex` folder.

```bash
# Navigate to affine-cortex directory
cd /root/veritas-rl/evaluate/affine-cortex

# Create virtual environment using uv (must be in affine-cortex folder)
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# Install affine-cortex dependencies
uv pip install -e .

# Install script dependencies
cd ../scripts
uv pip install -r requirements.txt
```

**Note**: The virtual environment (`.venv`) must be in the `affine-cortex` folder because:
- The `affine` module needs to be importable from the submodule
- The script automatically adds `affine-cortex` to Python path if not installed
- Both scripts and affine-cortex need access to the same dependencies

### 3. Environment Variables

Create a `.env` file in the project root (`/root/veritas-rl/.env`):

```bash
# Required for Chutes service (if using --uid)
CHUTES_API_KEY=your-api-key-here

# Optional: For W&B logging (recommended for tracking evaluations)
WANDB_API_KEY=your-wandb-key-here
```

**Important**: 
- The script automatically loads both `CHUTES_API_KEY` and `WANDB_API_KEY` from the `.env` file
- The `.env` file must be in the project root: `/root/veritas-rl/.env`
- The script searches for `.env` in this order:
  1. Project root (`/root/veritas-rl/.env`) - **Recommended location**
  2. Script directory (`/root/veritas-rl/evaluate/scripts/.env`)
  3. Current working directory (`.env`)

## Model Deployment (Required Before Evaluation)

**Important**: Before running evaluations, you must first deploy your model to Chutes using the benchmark tools. The evaluation script connects to a model service that must be running.

### Step 1: Deploy Model to Chutes

The model deployment is handled by the `benchmark` folder tools:

#### 1.1 Configure Chute Deployment

Edit `/root/veritas-rl/evaluate/benchmark/local_chute.py`:

```python
chute = build_sglang_chute(
    username="<your-chutes-username>",  # Replace with your Chutes username
    readme="<model-readme>",             # Model readme/repo
    model_name="<model-name>",           # Your model name
    image="chutes/sglang:0.5.1.post3",   # SGLang image
    concurrency=16,                      # Concurrent requests
    revision="<commit-hash>",            # Model revision/commit
    node_selector=NodeSelector(
        gpu_count=1,
        include=["h200"],                # GPU requirements
    ),
)
```

#### 1.2 Launch Chute Server

Deploy the model service using the launch script:

```bash
cd /root/veritas-rl/evaluate/benchmark
./local_launch.sh
```

This will:
- Start a Docker container with the SGLang image
- Deploy your model as a Chute
- Expose the service on port 30000
- Make it accessible at `http://localhost:30000/v1` (or `http://172.17.0.1:30000/v1` from containers)

**Note**: The `local_launch.sh` script runs the chute in development mode (`--dev`) locally. The service will be available as long as the container is running.

#### 1.3 Verify Deployment

Test that the service is running:

```bash
# From host
curl http://localhost:30000/v1/models

# From Docker container (if evaluating from container)
curl http://172.17.0.1:30000/v1/models
```

You should see a JSON response with model information.

### Step 2: Run Evaluation

Once the model service is deployed and running, proceed to the [Usage](#usage) section below.

## Model Service Connection

The evaluation script connects to the model service running on port 30000. The service should be accessible at `http://172.17.0.1:30000/v1` (Docker bridge gateway).

### Recommended Base URL

**Use `http://172.17.0.1:30000/v1`** when running evaluations, especially if:
- The evaluation runs inside a Docker container
- The model service is running on the host machine (from `local_launch.sh`)
- You're using Docker networking

This is the Docker bridge gateway IP that allows containers to access services on the host.

### Alternative Base URLs

- `http://host.docker.internal:30000/v1` - For Docker Desktop
- `http://localhost:30000/v1` - Only works if NOT running in Docker
- `http://<host-ip>:30000/v1` - Use your host machine's actual IP

## Usage

### Basic Evaluation

```bash
# Activate the virtual environment (from affine-cortex folder)
cd /root/veritas-rl/evaluate/affine-cortex
source .venv/bin/activate

# Navigate to scripts folder
cd ../scripts

# Evaluate using task IDs from task_ids.json (recommended)
./evaluate_local_model.py --env LGC-V2 --base-url http://172.17.0.1:30000/v1 --use-task-ids

# With output file
./evaluate_local_model.py --env LGC-V2 --base-url http://172.17.0.1:30000/v1 --use-task-ids --output results.json
```

### Evaluation Methods

#### 1. Using Task IDs from File (Most Common)

```bash
# Evaluate all tasks from task_ids.json
./evaluate_local_model.py --env PRINT --base-url http://172.17.0.1:30000/v1 --use-task-ids

# With custom output
./evaluate_local_model.py --env ABD --base-url http://172.17.0.1:30000/v1 --use-task-ids --output my_results.json
```

#### 2. Single Task Evaluation

```bash
# Single task, single sample
./evaluate_local_model.py --env ABD --base-url http://172.17.0.1:30000/v1 --task-id 5

# Single task, multiple samples
./evaluate_local_model.py --env ABD --base-url http://172.17.0.1:30000/v1 --task-id 5 --samples 10

# With specific seed
./evaluate_local_model.py --env ABD --base-url http://172.17.0.1:30000/v1 --task-id 5 --seed 42 --samples 5
```

#### 3. Range of Task IDs

```bash
# Evaluate tasks 0 through 9 (one sample per task)
./evaluate_local_model.py --env PRINT --base-url http://172.17.0.1:30000/v1 --task-id-range 0 9

# With output file
./evaluate_local_model.py --env ABD --base-url http://172.17.0.1:30000/v1 --task-id-range 0 4 --output range_results.json
```

#### 4. Using Chutes Service

```bash
# Requires CHUTES_API_KEY in .env file
./evaluate_local_model.py --env ABD-V2 --uid 7
```

## Command Line Arguments

### Required Arguments

- `--env` or `--envname`: Environment name (case-insensitive)
  - Supported: `SAT`, `ABD`, `ABD-V2`, `DED`, `DED-V2`, `LGC`, `SWE_PRO`, `CDE`, `GAME`, `PRINT`, `LGC-V2`

### Evaluation Method (choose one)

- `--task-id <id>`: Single task ID evaluation
- `--task-id-range <start> <end>`: Range of task IDs (one sample per task)
- `--use-task-ids`: Load task IDs from `task_ids.json` file

### Optional Arguments

- `--base-url <url>`: Model service URL
  - **Recommended**: `http://172.17.0.1:30000/v1` (for Docker environments)
  - Default: `http://localhost:30000/v1`
  
- `--samples <n>`: Number of samples per task (default: 1, only for `--task-id`)
- `--seed <n>`: Random seed (only for `--task-id`)
- `--temperature <float>`: Sampling temperature (default: 0.0)
- `--output <file>` or `-o <file>`: Save results to JSON file
- `--wandb-project <name>`: W&B project name (default: `{env}-Benchmark`)
- `--wandb-name <name>`: W&B run name (default: auto-generated)
- `--no-wandb`: Disable W&B logging

## Output Files

### results.json

Complete evaluation summary with:
- Total scores and averages
- Individual task results
- Error information (concise format)
- Configuration details

### rollout.jsonl

Line-delimited JSON file with individual evaluation results. Each line is a JSON object containing:
- Task ID and seed
- Score and success status
- Latency information
- Full conversation and metadata
- Error information (if failed)

## Troubleshooting

### Connection Errors

If you see `APIConnectionError: Connection error`:

1. **Check if chute server is running**:
   ```bash
   # Check if the chute-server container is running
   docker ps | grep chute-server
   
   # Test the service
   curl http://172.17.0.1:30000/v1/models
   ```

2. **If service is not running, deploy it first**:
   ```bash
   cd /root/veritas-rl/evaluate/benchmark
   ./local_launch.sh
   ```

3. **Verify Docker networking**:
   - If running in Docker, use `http://172.17.0.1:30000/v1` instead of `localhost`
   - Check if the service container is running: `docker ps | grep 30000`

4. **Test connection from container**:
   ```bash
   # If you're in a container
   curl http://172.17.0.1:30000/v1/models
   ```

### Chute Server Not Starting

If `local_launch.sh` fails:

1. **Check Docker is running**: `docker ps`
2. **Check port 30000 is available**: `netstat -tlnp | grep 30000`
3. **Check GPU availability**: `nvidia-smi` (if using GPU)
4. **Verify `local_chute.py` is configured correctly** with your model details
5. **Check Docker image exists**: `docker images | grep sglang`

### Module Not Found: affine

If you get `ModuleNotFoundError: No module named 'affine'`:

1. Ensure the virtual environment is activated
2. Verify the venv is in `affine-cortex` folder: `ls /root/veritas-rl/evaluate/affine-cortex/.venv`
3. Reinstall affine: `cd affine-cortex && uv pip install -e .`

### CHUTES_API_KEY Not Found

1. Create `.env` file in project root: `/root/veritas-rl/.env`
2. Add: `CHUTES_API_KEY=your-key-here`
3. The script automatically loads from project root

## Project Structure

```
veritas-rl/
├── evaluate/
│   ├── affine-cortex/          # Git submodule
│   │   ├── .venv/              # Virtual environment (MUST be here)
│   │   ├── affine/             # Affine module
│   │   └── pyproject.toml
│   ├── benchmark/               # Model deployment tools
│   │   ├── local_chute.py      # Chute configuration (edit this)
│   │   ├── local_launch.sh     # Launch chute server
│   │   └── affinetes_benchmark.py  # Alternative benchmark script
│   └── scripts/                # Evaluation scripts
│       ├── evaluate_local_model.py
│       ├── requirements.txt
│       ├── task_ids.json
│       └── README.md           # This file
└── .env                        # Environment variables (CHUTES_API_KEY, WANDB_API_KEY)
```

## Workflow Summary

1. **Deploy Model** (`benchmark/` folder):
   - Edit `local_chute.py` with your model details
   - Run `./local_launch.sh` to deploy model service on port 30000

2. **Evaluate Model** (`scripts/` folder):
   - Use `evaluate_local_model.py` to run evaluations
   - Connect to deployed service at `http://172.17.0.1:30000/v1`
   - Results saved to `results.json` and `rollout.jsonl`

## Examples

### Complete Workflow

```bash
# 1. Setup (one-time)
cd /root/veritas-rl
git submodule update --init --recursive
cd evaluate/affine-cortex
uv venv
source .venv/bin/activate
uv pip install -e .
cd ../scripts
uv pip install -r requirements.txt

# 2. Deploy model to Chutes (REQUIRED before evaluation)
cd ../benchmark
# Edit local_chute.py with your model details first!
./local_launch.sh
# Wait for service to start, then verify:
curl http://localhost:30000/v1/models

# 3. Run evaluation (in another terminal or after service is running)
cd ../scripts
source ../affine-cortex/.venv/bin/activate  # Activate venv if needed
./evaluate_local_model.py \
  --env LGC-V2 \
  --base-url http://172.17.0.1:30000/v1 \
  --use-task-ids \
  --output results.json

# 4. Check results
cat results.json | jq '.average_score'
```

### Testing Single Task

```bash
# Quick test before running full evaluation
./evaluate_local_model.py \
  --env PRINT \
  --base-url http://172.17.0.1:30000/v1 \
  --task-id 485649731 \
  --samples 1
```

## Benchmark Folder

The `benchmark` folder contains tools for deploying models to Chutes:

### Files

- **`local_chute.py`**: Chute configuration file
  - Defines model deployment settings (username, model name, image, GPU requirements)
  - Must be edited with your model details before deployment
  
- **`local_launch.sh`**: Launch script for local Chute deployment
  - Runs Docker container with SGLang
  - Deploys model as Chute service on port 30000
  - Service accessible at `http://localhost:30000/v1` (or `http://172.17.0.1:30000/v1` from containers)

- **`affinetes_benchmark.py`**: Alternative benchmark script
  - Runs evaluations across multiple environments
  - Uses Chutes cloud deployment (different from local deployment)

### Deployment Workflow

1. **Edit `local_chute.py`**:
   ```python
   chute = build_sglang_chute(
       username="your-username",
       model_name="your-model-name",
       # ... other settings
   )
   ```

2. **Launch the service**:
   ```bash
   cd /root/veritas-rl/evaluate/benchmark
   ./local_launch.sh
   ```

3. **Verify it's running**:
   ```bash
   curl http://localhost:30000/v1/models
   ```

4. **Run evaluation** (using `evaluate_local_model.py`):
   ```bash
   cd ../scripts
   ./evaluate_local_model.py --env LGC-V2 --base-url http://172.17.0.1:30000/v1 --use-task-ids
   ```

### Important Notes

- The chute server must be running before starting evaluation
- The service runs in a Docker container named `chute-server`
- Port 30000 must be available (or change it in `local_launch.sh`)
- The model service persists as long as the Docker container is running

## Quick Reference

### Recommended Command (Most Common Use Case)

```bash
# Activate venv (from affine-cortex folder)
cd /root/veritas-rl/evaluate/affine-cortex
source .venv/bin/activate

# Run evaluation
cd ../scripts
./evaluate_local_model.py \
  --env LGC-V2 \
  --base-url http://172.17.0.1:30000/v1 \
  --use-task-ids \
  --output results.json
```

### Key Points

- **Base URL**: Always use `http://172.17.0.1:30000/v1` for Docker environments
- **Virtual Environment**: Must be in `affine-cortex/.venv` (shared with scripts)
- **Submodule**: `affine-cortex` is a git submodule at `evaluate/affine-cortex`
- **Setup Tool**: Use `uv venv` and `uv pip install` for dependency management

## Notes

- The script automatically detects if it's running in Docker and suggests using `172.17.0.1` instead of `localhost`
- Results are saved incrementally to `rollout.jsonl` during evaluation
- Connection errors are logged concisely (error type + first line of message)
- The script checks connectivity before starting evaluation
- The virtual environment location is critical - it must be in `affine-cortex` folder for the script to find the `affine` module

