# AI-Powered Video Recreation System

Takes an existing video, extracts its content, and recreates it with new AI-generated visuals and voice-overs. Change the speaker's voice, alter their appearance, or modify the artistic style—all while preserving the original narrative and scene structure.

## Features

- **5-stage pipeline**: Analysis → Transcription → Audio Recreation → Video Generation → Assembly
- **Multiple video modes**:
  - **OmniHuman v1.5** (default) – Full-body talking video with lip sync, gestures, and high quality
  - **SadTalker** – Free, face-only talking head
  - **Scene-based** – Per-scene generation via Luma Dream Machine (no actor image needed)
- **Automatic text removal** – OCR + inpainting to strip captions/watermarks from actor images
- **Incremental processing** – Skips phases when outputs already exist
- **Cost tracking** – Saves API usage and estimated costs locally (ElevenLabs, Fal.ai)

## Prerequisites

- Python 3.10+
- FFmpeg (for audio/video processing)
- API keys for [ElevenLabs](https://elevenlabs.io/) and [Fal.ai](https://fal.ai/)

## Installation

```bash
# Clone or navigate to the project
cd video-to-video

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Setup

Create a `.env` file in the project root:

```env
ELEVENLABS_API_KEY=your_elevenlabs_api_key
FAL_API_KEY=your_fal_api_key
VOICE_ID=your_elevenlabs_voice_id  # Optional; default voice used if omitted
```

Get your keys from:
- [ElevenLabs Dashboard](https://elevenlabs.io/) – for text-to-speech
- [Fal.ai Dashboard](https://fal.ai/dashboard) – for video generation

## Usage

### OmniHuman v1.5 (default, best quality)

Creates a full-body talking video from an actor image and the source video's transcript:

```bash
python recreate_video.py Videos/your_video.mp4 --actor-image new-actor.jpeg
```

Options:
- `--resolution 720p|1080p` – 720p supports up to 60s audio, 1080p up to 30s
- `--turbo` – Faster generation (may reduce quality)

### SadTalker (free, face-only)

```bash
python recreate_video.py Videos/your_video.mp4 --mode sadtalker --actor-image actor.jpg
```

### Scene-based (Luma Dream Machine)

Generates a new clip per scene; no actor image required:

```bash
python recreate_video.py Videos/your_video.mp4 --mode scenes
```

With custom prompt template:

```bash
python recreate_video.py Videos/your_video.mp4 --mode scenes --prompt "A futuristic scene. {scene_text}"
```

### Other options

| Flag | Description |
|------|-------------|
| `--voice-id ID` | ElevenLabs voice ID (overrides `VOICE_ID` env) |
| `--whisper-model tiny\|base\|small\|medium\|large` | Whisper model size (default: base) |
| `--output-dir PATH` | Output directory (default: `<video_name>_v1`, `_v2`, …) |
| `--no-clean-image` | Skip text/caption removal from actor image |
| `--skip-video-gen` | Stop after audio generation (for testing) |

## Pipeline Overview

| Phase | Input | Output | Technology |
|-------|-------|--------|------------|
| 1. Analysis | Video file | `audio.mp3`, keyframes (`keyframes/*.jpg`) | MoviePy, PySceneDetect |
| 2. Transcription | Audio | `transcript.json` | OpenAI Whisper |
| 3. Audio Recreation | Transcript + voice ID | `new_audio.mp3` | ElevenLabs |
| 4. Video Generation | Keyframes / actor image + audio | Generated clips | Fal.ai (OmniHuman, SadTalker, Luma) |
| 5. Assembly | Clips + new audio | `final_video.mp4` | MoviePy |

Outputs are written to `<video_name>_v1/`, `<video_name>_v2/`, etc., under the video’s parent directory.

## Cost Tracking

Each run records API usage and estimated costs to:

- **`costs.json`** (project root) – Cumulative log of all runs with totals
- **`<output_dir>/costs.json`** – Per-run breakdown for that output

Tracked services:
- **ElevenLabs** – Characters used for TTS (~$0.12 per 1K chars)
- **Fal.ai** – Per-call estimates for OCR, FLUX inpainting, OmniHuman, SadTalker, Luma Dream Machine

Pricing is configurable in `cost_tracker.py` (`DEFAULT_PRICING`). Actual billing may differ by account.

## Project Structure

```
video-to-video/
├── recreate_video.py    # Main pipeline script
├── cost_tracker.py       # API usage & cost tracking
├── costs.json            # Cumulative cost log (generated, gitignored)
├── requirements.txt
├── .env                  # API keys (create from template above)
├── PRD/                  # Product requirements & guides
└── Videos/               # Source videos and outputs
```

## References

- [MoviePy](https://zulko.github.io/moviepy/)
- [PySceneDetect](https://www.scenedetect.com/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [ElevenLabs API](https://elevenlabs.io/docs/api-reference/introduction)
- [Fal.ai Documentation](https://docs.fal.ai/)
