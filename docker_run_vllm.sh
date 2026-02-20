docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --served-model-name my-model \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct
    -- enable-auto-tool-choice \
    --tool-call-parser hermes
