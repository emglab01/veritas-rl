SHELL := /bin/bash

.PHONY: help setup uv-install docker-train sft-train rl-train merge-checkpoints eval-local

help:
	@echo "Available targets:"
	@echo "  setup             - Run Python environment setup via setup_uv.sh"
	@echo "  uv-install        - Install Python packages with uv (editable mode)"
	@echo "  docker-train      - Start VERL Docker training container"
	@echo "  sft-train         - Run LGC-V2 SFT training script"
	@echo "  rl-train          - Run LGC-V2 RL GRPO training script"
	@echo "  merge-checkpoints - Merge LoRA / checkpoints via merge.sh"
	@echo "  eval-local        - Launch local chute model for evaluation"

setup:
	chmod +x ./setup_uv.sh
	./setup_uv.sh

uv-install:
	uv pip install -e .

docker-train:
	cd train && chmod +x ./docker_run_verl.sh && ./docker_run_verl.sh

sft-train:
	docker exec -it verl bash -lc 'cd /workspace/veritas-rl && chmod +x ./train/scripts/lgc-v2-SFT-trainer.sh && ./train/scripts/lgc-v2-SFT-trainer.sh'

rl-train:
	docker exec -it verl bash -lc 'cd /workspace/veritas-rl && chmod +x ./train/scripts/lgc-v2-RL-GRPO-trainer.sh && ./train/scripts/lgc-v2-RL-GRPO-trainer.sh'

merge-checkpoints:
	docker exec -it verl bash -lc 'cd /workspace/veritas-rl && chmod +x ./train/scripts/merge.sh && ./train/scripts/merge.sh'

eval-local:
	chmod +x ./evaluate/local_chute_deploy_model/local_launch.sh
	./evaluate/local_chute_deploy_model/local_launch.sh


