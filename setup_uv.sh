#!/bin/bash
# Quick setup script for veritas-rl using uv
# This sets up the environment for data processing and evaluation scripts.
# For training, use Docker instead (see train/docker_run_verl.sh).

set -e

echo "🚀 Setting up veritas-rl with uv..."
echo "📝 Note: This setup is for data processing/evaluation scripts."
echo "   For training, use Docker: ./train/docker_run_verl.sh"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed."
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed. Please restart your shell or run: source ~/.cargo/env"
    exit 1
fi

echo "✅ uv is installed"

# Create virtual environment
echo "📦 Creating virtual environment..."
uv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
uv pip install -e .

# Set up .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from env.example..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your API keys:"
    echo "   - OPENAI_API_KEY (for add_reasoning.py)"
    echo "   - HF_TOKEN (for Trace environment)"
    echo "   - WANDB_API_KEY (optional, for training)"
    echo "   - CHUTES_API_KEY (optional, for env evaluation)"
else
    echo "✅ .env file already exists"
fi

# Verify installation
echo "🔍 Verifying installation..."
python -c "import torch; import transformers; import openai; import datasets; print('✅ All core dependencies installed')" || {
    echo "❌ Some dependencies are missing. Try: uv pip install -e . --force-reinstall"
    exit 1
}

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Next steps:"
echo "   1. Edit .env file with your API keys"
echo "   2. Activate the virtual environment: source .venv/bin/activate"
echo "   3. Run data processing scripts (see README.md)"
echo ""
echo "📖 For more information, see:"
echo "   - docs/setup_uv.md - Detailed uv setup guide"
echo "   - docs/dataset_generation.md - Dataset generation guide"
echo "   - README.md - Project overview"
echo ""

