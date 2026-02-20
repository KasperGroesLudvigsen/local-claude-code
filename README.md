# local-claude-code

We will use vLLM to run a self-hosted model and connect Claude Code to it. 

# TODO

- Build vLLM docker image that is compatible with the Blackwell architecture. See this: https://discuss.vllm.ai/t/support-for-rtx-6000-blackwell-96gb-card/1707/4 . STATUS 20260211: Failed to build. Could not resolve the dependencies in requirements/build.txt. Will now try one of the images mentioned here: https://forums.developer.nvidia.com/t/new-pre-built-vllm-docker-images-for-nvidia-dgx-spark/357832/9 

- Consider using Nvidia TRT LLM for inference if vLLM proves too cumbersome to build https://build.nvidia.com/spark/trt-llm/instructions 

## Set up local LLM with Ahtropic-compatible endpoint

You need to expose an LLM via an Anthropic compatible API endpoint. I recommend using vLLM for that. 

You can run vLLM directly on your OS, or inside a Docker container. 

To run it on your OS, you need to install vLLM first using the [official installation guide](https://docs.vllm.ai/en/latest/getting_started/quickstart/) or install Docker. 


Install [Claude Code](https://code.claude.com/docs/en/setup)

Source: https://docs.vllm.ai/en/latest/serving/integrations/claude_code/ 


# Test bare minimum

Install vLLM using `uv`

Run:
```bash
vllm serve openai/gpt-oss-120b --served-model-name my-model --enable-auto-tool-choice --tool-call-parser openai
```

If you get an error like this one below, it means your CUDA driver is 13.0, but vLLM is only compatible with CUDA < 13.0 as of 20260210.
```bash
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
```

To install CUDA=12.0, do this:
```bash
sudo apt install libcudart12
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
Now run the `vllm` command again. 


# Start Open WebUI
docker run -d -p 3000:8080 --gpus all --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:cuda



# Run docker without sudo
Sidenote: To be able to run Docker without `sudo`, do this: 
```bash
sudo usermod -aG docker $USER
```
And then restart your computer.
