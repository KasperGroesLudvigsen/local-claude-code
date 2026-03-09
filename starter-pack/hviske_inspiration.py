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
