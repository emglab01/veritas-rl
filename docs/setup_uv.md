# Setting Up veritas-rl with uv

This guide explains how to set up and run the veritas-rl project using `uv` (a fast Python package manager).

## When to Use This Setup

**This setup is for data processing and evaluation scripts** that run on your host machine:
- `data_processing/generate_dataset.py` - Generate datasets
- `data_processing/add_reasoning.py` - Add reasoning to datasets
- `data_processing/evaluate_dataset.py` - Evaluate datasets
- `data_processing/convert_to_parquet.py` - Convert to parquet format

**For training**, use Docker instead (see `train/docker_run_verl.sh`).

## Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Or on macOS/Linux:
   brew install uv
   # Or with pip:
   pip install uv
   ```

2. **Python 3.11 or 3.12** (required by the project)

## Quick Start

### 1. Create Virtual Environment with uv

```bash
cd /root/veritas-rl

# Create virtual environment
uv venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install all project dependencies
uv pip install -e .

# Or install from pyproject.toml directly
uv pip sync pyproject.toml
```

### 3. Verify Installation

```bash
# Check that key packages are installed
python -c "import torch; import transformers; import openai; import datasets; print('✓ All dependencies installed')"
```

## Project Structure

The project has dependencies organized by component:

### Core Dependencies (in pyproject.toml)

- **ML/AI**: `torch`, `transformers`, `datasets`, `peft`, `trl`
- **Data Processing**: `pandas`, `pyarrow`
- **APIs**: `openai`, `httpx`
- **Configuration**: `python-dotenv`, `omegaconf`, `pydantic`
- **Logging**: `colorlog`, `structlog`
- **System**: `psutil`

### Environment-Specific Dependencies

Some environments have additional requirements:

- **LGC-V2**: `openai`, `httpx`, `structlog`
- **Trace**: `datasets`, `openai`, `httpx`, `pydantic`, `psutil`, `structlog`

These are included in the main `pyproject.toml` dependencies.

## Usage Examples

### Running Data Processing Scripts

```bash
# Generate dataset
python data_processing/generate_dataset.py \
  --env lgc-v2 \
  --output data_processing/dataset/lgc-v2.jsonl \
  --num-samples 1000

# Add reasoning to dataset
python data_processing/add_reasoning.py \
  --input data_processing/dataset/lgc-v2.jsonl \
  --output data_processing/dataset/lgc-v2-with-reasoning.jsonl \
  --model gpt-4o-mini

# Evaluate dataset
python data_processing/evaluate_dataset.py \
  --input data_processing/dataset/lgc-v2.jsonl \
  --env lgc-v2 \
  --output data_processing/results/lgc-v2-eval.json

# Convert to parquet
python data_processing/convert_to_parquet.py \
  --input data_processing/dataset/lgc-v2.jsonl \
  --out-dir train/parquet \
  --env lgc-v2 \
  --format rl
```

### Setting Up Environment Variables

```bash
# Copy example env file
cp env.example .env

# Edit .env with your API keys
nano .env
```

Required environment variables:
- `OPENAI_API_KEY` - for `add_reasoning.py`
- `HF_TOKEN` - for Trace environment and model downloads
- `WANDB_API_KEY` - for training (optional)
- `CHUTES_API_KEY` - for environment evaluation (optional)

## Troubleshooting

### Issue: Import errors after installation

**Solution**: Make sure you're in the virtual environment:
```bash
# Check if venv is activated (should show .venv in prompt)
which python  # Should point to .venv/bin/python

# If not activated:
source .venv/bin/activate
```

### Issue: Missing dependencies

**Solution**: Reinstall dependencies:
```bash
uv pip install -e . --force-reinstall
```

### Issue: CUDA/GPU not available

**Solution**: Install PyTorch with CUDA support:
```bash
# Check CUDA version
nvidia-smi

# Install appropriate PyTorch version
# For CUDA 11.8:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Environment scripts can't find modules

**Solution**: The environment scripts use relative imports. Make sure you run from the project root:
```bash
cd /root/veritas-rl
python data_processing/generate_dataset.py ...
```

## Advanced: Using uv for Development

### Install with Development Dependencies

```bash
uv pip install -e ".[dev]"
```

### Update Dependencies

```bash
# Update all dependencies to latest compatible versions
uv pip install --upgrade -e .

# Update specific package
uv pip install --upgrade openai
```

### Lock Dependencies

```bash
# Generate/update uv.lock file
uv lock
```

## Comparison with Other Package Managers

| Feature | uv | pip | conda |
|---------|----|-----|-------|
| Speed | ⚡ Very Fast | 🐌 Slow | 🐌 Slow |
| Virtual Env | ✅ Built-in | ✅ Built-in | ✅ Built-in |
| Lock Files | ✅ uv.lock | ❌ No | ✅ environment.yml |
| Python Management | ✅ Yes | ❌ No | ✅ Yes |

## Next Steps

1. **Set up `.env` file** with your API keys (see `env.example`)
2. **Generate a test dataset** to verify everything works
3. **Read the documentation**:
   - `docs/dataset_generation.md` - Dataset generation guide
   - `docs/env_file_implementation.md` - Environment variables guide
   - `README.md` - Project overview

## Additional Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [Project README](README.md)
- [Dataset Generation Guide](docs/dataset_generation.md)

