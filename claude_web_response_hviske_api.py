"""
Prompt:
`Write a Python FastAPI endpoint that uses the Hviske 3.0 Huggingface model to transcribe incoming audio files`

"""


import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

app = FastAPI(title="Hviske 3.0 Transcription API")

# --- Model Setup ---

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_ID = "alexandrainst/hviske-3.0"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
).to(device)

transcription_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# --- Endpoints ---

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "danish",
    return_timestamps: bool = False,
):
    """Transcribe an audio file using Hviske 3.0."""

    allowed_types = {
        "audio/mpeg", "audio/wav", "audio/x-wav",
        "audio/flac", "audio/ogg", "audio/webm",
    }
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. "
                   f"Accepted: {', '.join(allowed_types)}",
        )

    try:
        audio_bytes = await file.read()

        generate_kwargs = {"language": language, "task": "transcribe"}

        result = transcription_pipeline(
            io.BytesIO(audio_bytes),
            generate_kwargs=generate_kwargs,
            return_timestamps=return_timestamps,
        )

        response = {
            "filename": file.filename,
            "text": result["text"],
        }
        if return_timestamps and "chunks" in result:
            response["chunks"] = result["chunks"]

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "device": device}