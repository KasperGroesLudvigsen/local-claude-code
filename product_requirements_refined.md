# LOCAL TRANSCRIPTION API ENDPOINT

This project is about creating an API service that exposes a transcription model (speech to text). 

## Requirements
- Use FastAPI to expose a "/transcribe" endpoint.
- The service must be able to handle multiple requests simultaneously without crashing, e.g. via some kind of queue either explicitly or implicitly.
- Run the service inside one or several Docker containers based on the complexity of the system. Default to a modular microservice architecture if needed.  
- Launch Docker container via Docker compose
- Must be compatible with the CUDA information described in the section "nvidia-smi output". 
- MUST use this hugging face: model_id = "syvai/hviske-v3-conversation". See "hviske_inspiration.py" for an example of how to use the model given in the model card on huggingface. 
- The /transcribe endpoint must receive an audio file and return transcriptions. If the underlying transcription model supports it, the following must be returned: 1) A full text transcription, 2) time stamped transcriptions. 
- If the underlying model supports it, the service must include a "detect language" endpoint
- The whole service including model inference and API endpoints will be running on a Fusionxpark with 128GB unified VRAM. 
- The service is primarily intended for transcription of Danish speech. 
- Allow both long and big audio files. Set liberal limits. 
- We expect low to moderate load on the service. We expect no more than 10 simultaneous requests at any time. 



## `nvidia-smi` output
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0  On |                  N/A |
| N/A   46C    P0             12W /  N/A  | Not Supported          |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3038      C   /usr/local/bin/python3                  274MiB |
|    0   N/A  N/A            3981      G   /usr/lib/xorg/Xorg                      207MiB |
|    0   N/A  N/A            4116      G   /usr/bin/gnome-shell                    163MiB |
|    0   N/A  N/A            5755      G   /usr/share/code/code                     90MiB |
|    0   N/A  N/A            8449      G   .../7832/usr/lib/firefox/firefox        228MiB |
+-----------------------------------------------------------------------------------------+


# LOCAL TRANSCRIPTION API ENDPOINT

## Project Overview
This project involves creating an API service that exposes a speech-to-text transcription model. The service will be built using FastAPI and deployed in a Dockerized environment.

## Business Context
- **Primary Use Case**: Transcription of Danish speech
- **Target Environment**: Fusionxpark with 128GB unified VRAM
- **Expected Load**: Low to moderate (max 10 simultaneous requests)
- **Performance Requirements**: Handle large audio files with liberal size limits

## Technical Requirements

### Core Architecture
- **Framework**: FastAPI for API endpoint creation
- **Deployment**: Docker containers with Docker Compose orchestration
- **Microservices**: Modular architecture (default approach)
- **GPU Compatibility**: Must work with CUDA 13.0 and driver version 580.126.09

### Model Specifications
- **Primary Model**: "syvai/hviske-v3-conversation" Hugging Face Transformers
- **Model Focus**: Danish speech transcription
- **Model Capabilities**:
  - Full text transcription
  - Time-stamped transcriptions (when supported)
  - Language detection capability (when supported)

### API Endpoints
#### Required Endpoint
- `/transcribe` - Accepts audio files and returns transcriptions
  - Input: Audio file (support large/bulk files)
  - Output: 
    - Full text transcription
    - Time-stamped transcriptions (if model supports)
  - Error handling for various audio formats and sizes

#### Optional Endpoint
- `/detect_language` - Language detection endpoint (if model supports)
  - Input: Audio file
  - Output: Detected language information

### Performance & Scalability
- **Concurrency**: Handle up to 10 simultaneous requests without crashing
- **Queue Management**: Implement queuing mechanism (explicit or implicit)
- **Resource Utilization**: Optimize for 128GB unified VRAM capacity
- **File Size Support**: Liberal limits for audio file processing

### Hardware & Environment
- **GPU Information**: 
  - Device: NVIDIA GB10
  - Driver: 580.126.09
  - CUDA: 13.0
  - Memory: Not Supported (VRAM utilization managed by application)
- **System**: Fusionxpark with 128GB unified VRAM

## Implementation Details

### Development Approach
- Use `hviske_inspiration.py` for an example of how to use the model given in the model card on huggingface. Use as reference implementation
- Follow modular microservice design principles
- Ensure robust error handling and logging
- Implement proper resource management for GPU memory

### File Handling
- Support thse audio formats: mp3, wav, flac, m4a/aac
- Check if the file has a compatible format
- Handle large audio files efficiently
- Implement streaming or chunked processing when necessary
- Max file size limit: 500 MB
- Make a differentiated timeout strategy, allowing longer files a longer time out thresholds. Expect processing of large files to take up to 30 minutes

## Success Criteria
- API endpoint responds correctly to transcription requests
- Supports concurrent processing up to 10 requests
- Utilizes GPU resources effectively within 128GB VRAM limit
- Provides accurate Danish speech transcription
- Maintains system stability under expected load conditions

## Constraints
- Must be compatible with NVIDIA driver 580.126.09 and CUDA 13.0
- Must support Hviske 2.0 model from Hugging Face Transformers
- Maximum concurrent requests: 10
- Primary focus on Danish language transcription