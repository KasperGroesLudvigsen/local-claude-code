# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI-based transcription service that exposes a speech-to-text model (specifically "syvai/hviske-v3-conversation") through API endpoints. The service is designed to run on a system with NVIDIA GPU (specifically NVIDIA GB10 with CUDA 13.0) and is optimized for Danish speech transcription.

## Key Components

- **FastAPI Application**: Main API service exposing `/transcribe` and optional `/detect_language` endpoints
- **Hugging Face Model Integration**: Uses "syvai/hviske-v3-conversation" for speech-to-text conversion
- **Docker Deployment**: Containerized with Docker Compose for orchestration
- **GPU Memory Management**: Optimized for 128GB unified VRAM environment

## Development Setup

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- NVIDIA GPU with CUDA 13.0 support
- 128GB unified VRAM (Fusionxpark environment)

### Running the Service
```bash
# Start the service with Docker Compose
docker-compose up

# Run tests
pytest tests/

# Lint code
ruff check .
ruff format .
```

### Testing Individual Components
```bash
# Run a specific test
pytest tests/test_transcribe_endpoint.py::test_valid_audio_file

# Test with coverage
pytest tests/ --cov=src/
```

## Architecture

### API Endpoints
- `/transcribe` - POST endpoint for audio file transcription
- `/detect_language` - Optional endpoint for language detection (if model supports)

## Implementation Guidelines

1. **Model Usage**: Refer to `hviske_inspiration.py` for model loading patterns
2. **Concurrency**: Support up to 10 simultaneous requests with proper queuing
3. **GPU Memory**: Manage VRAM efficiently for large audio files
4. **Error Handling**: Robust handling of invalid audio formats, sizes, and processing errors
5. **File Support**: Handle common audio formats (mp3, wav, flac, m4a/aac) with 500MB max file size

## Docker Deployment

The service should be deployable using Docker Compose with GPU support enabled. Ensure the container has access to the GPU and sufficient VRAM allocation.

## Performance Considerations

- Process large audio files with streaming or chunked processing
- Implement different timeout strategies for large files (up to 30 minutes)
- Monitor GPU memory usage to prevent out-of-memory errors
- Handle concurrent requests efficiently without system crashes

## Testing Strategy

- Unit tests for individual components
- Integration tests for API endpoints
- Load testing for concurrent request handling
- GPU memory usage tests
- Audio format validation tests

## Code Quality Standards

- Follow FastAPI best practices
- Maintain minimal code impact
- Ensure GPU memory is properly managed
- Implement proper logging and error reporting
- Follow existing project conventions and patterns


## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for **ANY non-trivial task** (3+ steps or architectural decisions)
- If something goes sideways, **STOP and re-plan immediately** — don’t keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop

- After **ANY correction from the user**: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: **“Would a staff engineer approve this?”**
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask **“Is there a more elegant way?”**
- If a fix feels hacky:  
  *“Knowing everything I know now, implement the elegant solution”*
- Skip this for simple, obvious fixes — don’t over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing

- When given a bug report: **just fix it**. Don’t ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

## Task Management

- **Plan First**: Write plan to `tasks/todo.md` with checkable items
- **Verify Plan**: Check in before starting implementation
- **Track Progress**: Mark items complete as you go
- **Explain Changes**: High-level summary at each step
- **Document Results**: Add review section to `tasks/todo.md`
- **Capture Lessons**: Update `tasks/lessons.md` after corrections

---

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Minimal code impact.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Touch only what’s necessary. Avoid introducing bugs.