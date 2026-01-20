export PYTHONPATH=/workspace/verl:
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir artifacts/checkpoints/Qwen3-4B-nntoan-20251103_174435-batch_size-128_max_length-32768/global_step_100/ \
    --target_dir merged_model