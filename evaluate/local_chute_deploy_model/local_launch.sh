IMAGE="lmsysorg/sglang:v0.5.3rc1-cu128-b200"
docker run -it \
    --gpus all \
    --name chute-server \
    -v ./local_chute.py:/sgl-workspace/sglang/local_chute.py:ro \
    -e CHUTES_EXECUTION_CONTEXT=REMOTE \
    -p 30000:30000 \
    $IMAGE \
    bash -c "pip install chutes==0.3.30 && chutes run local_chute:chute --port 30000 --dev"
