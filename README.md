# BLog post: Building a transcription endpoint with Claude Code running a local LLM


Set up: 
- Model Qwen/Qwen3-Coder-30B-A3B-Instruct
- Chat interface: OpenWebUI
- Make Claude Code use Qwen istead of Antrhopic native models
- FusionXpark 128 GB Vram

## Approach: 
1. Write requirements braindump - see `product_requirements_braindump.md`
2. Ask LLM for feedback through Open WebUI, resulting in `product_requirements_refined.md`
3. Save as "product_requirements.md"
3. Define CLAUDE.md
5. Enter /plan mode and get started with this prompt: "read the product_requirements.md file and let's plan how to build this project. Ask clarifying questions liberally."

## Stats Qwen3
Avg prefill speed: 10,000 - 15,000 tokens/s
Avg tokens generated per second (decoding): 10-20 tokens

Main code generation run: 10m 56s

Completing this prompt took 2m 7s: 
```
refactor test_api.py so that the audio file path is not hardcoded but provided as an argument from the command line when executing the script, like: python3 test_api.py my_audio.wav
```

## Learnings

### Using Qwen/Qwen3-Coder-30B-A3B-Instruct

It was done after N turns (one of them being I wanted to change the port from 8000 that VLLM was already using. I could have specified this but it could also have checked itself or asked)

Where it went wrong: 
1. Did not anticipate that a token could be required for model loading
1. First version of code was not written for GPU despite being specifically instructed to write for GPU deployment. After raising this issue, claude fixed it straight away. 
3. didnt install ffmpeg in docker image
4. RUnning out of context windoe: ● Background command "Rebuild the Docker image with FFmpeg installed" completed (exit code 0)                                                                          
  ⎿  API Error: 400 {"type":"error","error":{"type":"BadRequestError","message":"'max_tokens' or 'max_completion_tokens' is too large: 32000. This model's maximum     
     context length is 100001 tokens and your request has 81126 input tokens (32000 > 100001 - 81126). (parameter=max_tokens, value=32000)"}}  
5. then: "Error: Transcription failed: Transcription failed: Unable to create tensor, you should probably activate padding with 'padding=True' to have batched tensors with the same length." Took 5m 46s to refactor. 


More context on #5: Qwen had generated a requirements file requiring specific versions of the required pakcakge (using the `==` operator). This caused an imcompatibility issue between Numpy and Torch. Once that was solved, it worked. It had also for some reason isntalled a cpu-ony version of Torch, despite clear instructions in the `product_requirements.md` file. I discovered this myself and fixed. It's weird it did itself not wonder "why are we running on cpu" when I said "GPU" in several instructions. 

# Basic steps:

### Run local LLM

```bash
docker run -d --privileged --gpus all --rm --ipc=host --network host --name my-model \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  scitrera/dgx-spark-vllm:0.14.0-t5 \
  vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --gpu-memory-utilization 0.85 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --served-model-name my-model \
  --max-model-len 100001
```
or
```bash
docker compose -f deployment/compose.gpt-oss.yml up
```

### Set up Open WebUI (OPTIONAL)
Run OpenWebUI to have a chat interface in addition to Claude Code:
```bash
docker run -d -p 3000:8080 --gpus all --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:cuda
```

Inside Open WebUI, go to "Admin Panel" --> "Settings" --> "Connection" and add your local model as an OpenAI API. Insert this in "URL":

`http://host.docker.internal:8000/v1`

And enter a dummy Bearer token under "Auth", e.g."

`sk-dummy`

Click save and Open WebUI is good to go. 

### Run Claude Code
Open a terminal, navigate to the project you want to work on. Then, run:
```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_OPUS_MODEL=my-model \
ANTHROPIC_DEFAULT_SONNET_MODEL=my-model \
ANTHROPIC_DEFAULT_HAIKU_MODEL=my-model \
claude
```

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

## Candidate models
GLM-5 AWQ
