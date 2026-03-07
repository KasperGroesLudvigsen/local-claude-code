# LOCAL TRANSCRIPTION API ENDPOINT

This project is about creating an API service that exposes a transcription 
model (speech to text).

## Requirements

- Use FastAPI to expose a `/transcribe` endpoint.
- The service must be able to handle multiple requests simultaneously without 
  crashing. We expect no more than 10 simultaneous requests at any time.
- Run the service inside one or several Docker containers. Default to a modular 
  microservice architecture if the complexity warrants it.
- Launch Docker containers via Docker Compose.
- Must be compatible with the CUDA/hardware configuration described in the 
  "nvidia-smi output" section below.
- MUST use `syvai/hviske-v3-conversation` via HuggingFace Transformers.
- The `/transcribe` endpoint must receive an audio file and return:
  1. A full text transcription
  2. Timestamped transcription segments (word or segment level)
- The service is primarily intended for transcription of Danish speech. 
  Force `language="da"` — do not implement language auto-detection.
- Allow both long and large audio files. Set liberal file size and duration 
  limits.
- The whole service will run on a machine with 128GB unified VRAM (see 
  hardware section).


# Code example from syvai/hviske-v3-conversation model card

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

# Sæt device og data type for optimal performance
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Angiv model ID for Hviske v3
model_id = "syvai/hviske-v3-conversation"

# Hent model og processor fra Hugging Face
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Opret en ASR pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)

# Eksempel: Transskriber en lydfil fra CoRal datasættet
# Du kan erstatte dette med din egen lydfil: f.eks. pipe("sti/til/din/lydfil.wav")
dataset = load_dataset("alexandrainst/coral", split="test", streaming=True)
sample = next(iter(dataset))["audio"]

result = pipe(sample)
print(result["text"])
```
## Hardware & CUDA

+-----------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09   Driver Version: 580.126.09   CUDA Version: 13.0 |
+---------------------------+--------------------+------------------------+
| GPU  Name        Pers-M  | Bus-Id      Disp.A | Volatile Uncorr. ECC  |
|   0  NVIDIA GB10    On   | 0000000F:01:00.0   | N/A                   |
+---------------------------+--------------------+------------------------+

- GPU: NVIDIA GB10 (DGX Spark / FusionxPark), ARM-based architecture
- Unified memory: 128GB shared between CPU and GPU