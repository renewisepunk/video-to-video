#!/usr/bin/env python3
"""
AI-Powered Video Recreation System

Takes an existing video, extracts its content, and recreates it with
new AI-generated visuals and voice-overs.

Pipeline:
  1. Video Analysis   - Extract audio + detect scenes/keyframes
  2. Transcription    - Speech-to-text with timestamps (Whisper)
  3. Audio Recreation - Text-to-speech with new voice (ElevenLabs)
  4. Video Generation - Image-to-video for each scene (Fal.ai)
  5. Final Assembly   - Concatenate clips + attach new audio
"""

import os
import sys
import json
import glob
import argparse
import requests
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip
from scenedetect import detect, AdaptiveDetector, save_images, open_video
import whisper
from elevenlabs.client import ElevenLabs
import fal_client

from cost_tracker import CostTracker

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

FAL_API_KEY = os.getenv("FAL_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_VOICE_ID = os.getenv("VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

if FAL_API_KEY:
    os.environ["FAL_KEY"] = FAL_API_KEY


# ---------------------------------------------------------------------------
# Actor Image Generation (Google Gemini)
# ---------------------------------------------------------------------------

def _analyze_scene(keyframe_path):
    """Use Gemini Vision to describe the scene from a keyframe (no text/captions)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    image = Image.open(keyframe_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_image(image),
            "Describe this video frame for recreating it with a different person. "
            "Focus on: camera angle, framing (close-up/medium/wide), background "
            "environment, lighting (natural/studio/warm/cool), clothing style, "
            "body posture and hand gestures, overall mood. "
            "Ignore any text, captions, or watermarks. "
            "Be specific and concise (3-4 sentences).",
        ],
    )
    return response.text


def generate_actor_image(actor_description, keyframe_path, output_dir):
    """Generate an actor image matching the original video's style.

    1. Analyze a keyframe to extract scene context (background, lighting, framing)
    2. Combine with user's actor description to generate a coherent new image
    """
    print("\n=== Generating Actor Image (Gemini) ===")

    actor_path = os.path.join(output_dir, "generated_actor.png")
    if os.path.exists(actor_path):
        print(f"  Actor image already exists: {actor_path}")
        return actor_path

    from google import genai

    # Step 1: Analyze the original scene
    print("  Analyzing original video scene...")
    scene_description = _analyze_scene(keyframe_path)
    print(f"  Scene: {scene_description}")

    # Step 2: Build a coherent prompt combining scene + new actor
    prompt = (
        f"Generate a realistic photograph of the following person: {actor_description}. "
        f"Place them in this exact scene and style: {scene_description}. "
        "The image should look like a real video frame, not AI-generated. "
        "Portrait orientation. No text, no captions, no watermarks."
    )
    print(f"  Generating new actor image...")

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt],
    )

    for part in response.parts:
        if part.inline_data is not None:
            image = part.as_image()
            image.save(actor_path)
            print(f"  -> Actor image saved: {actor_path}")
            return actor_path

    print("ERROR: Gemini did not return an image.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Phase 1 – Video Analysis & Extraction
# ---------------------------------------------------------------------------

def analyze_video(video_path, output_dir):
    """Extract audio and detect scene keyframes from the source video."""
    print("\n=== Phase 1: Video Analysis ===")

    # --- Extract audio ---
    audio_path = os.path.join(output_dir, "audio.mp3")
    if not os.path.exists(audio_path):
        print("Extracting audio...")
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, logger=None)
        clip.close()
        print(f"  -> Saved audio to {audio_path}")
    else:
        print("Audio already extracted, skipping.")

    # --- Detect scenes and save keyframes ---
    keyframe_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(keyframe_dir, exist_ok=True)

    existing_keyframes = glob.glob(os.path.join(keyframe_dir, "*.jpg"))
    if not existing_keyframes:
        print("Detecting scenes and extracting keyframes...")
        scene_list = detect(video_path, AdaptiveDetector())

        if not scene_list:
            # Fallback: if no scene changes detected, grab a single frame
            print("  No scene changes detected; extracting a single keyframe.")
            clip = VideoFileClip(video_path)
            mid = clip.duration / 2
            frame_path = os.path.join(keyframe_dir, "001.jpg")
            clip.save_frame(frame_path, t=mid)
            clip.close()
        else:
            video_stream = open_video(video_path)
            save_images(
                scene_list=scene_list,
                video=video_stream,
                num_images=1,
                output_dir=keyframe_dir,
                image_name_template="$SCENE_NUMBER",
            )
        existing_keyframes = glob.glob(os.path.join(keyframe_dir, "*.jpg"))
        print(f"  -> Extracted {len(existing_keyframes)} keyframe(s)")
    else:
        print(f"Keyframes already extracted ({len(existing_keyframes)} found), skipping.")

    return audio_path, keyframe_dir


# ---------------------------------------------------------------------------
# Phase 2 – Transcription
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path, output_dir, whisper_model="base"):
    """Transcribe the audio file using OpenAI Whisper."""
    print("\n=== Phase 2: Transcription ===")

    transcript_path = os.path.join(output_dir, "transcript.json")
    if not os.path.exists(transcript_path):
        print(f"Transcribing audio (model={whisper_model})...")
        model = whisper.load_model(whisper_model)
        result = model.transcribe(audio_path)
        with open(transcript_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> Transcript saved to {transcript_path}")
        return result
    else:
        print("Transcript already exists, loading.")
        with open(transcript_path, "r") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Phase 3 – Audio Recreation (ElevenLabs TTS)
# ---------------------------------------------------------------------------

def recreate_audio(transcript, voice_id, output_dir, cost_tracker=None):
    """Generate a new audio track from the transcript using ElevenLabs."""
    print("\n=== Phase 3: Audio Recreation ===")

    new_audio_path = os.path.join(output_dir, "new_audio.mp3")
    if not os.path.exists(new_audio_path):
        text = transcript["text"]
        print(f"Generating speech with voice_id={voice_id}...")
        from elevenlabs import VoiceSettings

        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_v3",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.0,              # "Creative" - most expressive/natural
                similarity_boost=0.78,
                style=0.0,
                use_speaker_boost=False,
            ),
        )
        with open(new_audio_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)
        print(f"  -> Raw TTS audio saved to {new_audio_path}")

        if cost_tracker:
            cost_tracker.record_elevenlabs(len(text), model_id="eleven_v3")

        # Light post-processing for naturalness
        _post_process_audio(new_audio_path)
    else:
        print("New audio already generated, skipping.")

    return new_audio_path


def _post_process_audio(audio_path):
    """Apply light post-processing: high-pass filter + loudness normalization."""
    try:
        from pydub import AudioSegment
        from pydub.effects import normalize, high_pass_filter

        print("  Applying audio post-processing...")
        audio = AudioSegment.from_mp3(audio_path)

        # Remove sub-80Hz rumble common in TTS output
        audio = high_pass_filter(audio, cutoff=80)

        # Normalize loudness (~-16 LUFS broadcast level)
        audio = normalize(audio, headroom=1.0)

        audio.export(audio_path, format="mp3", bitrate="128k")
        print("  -> Post-processing applied (high-pass + normalization)")
    except ImportError:
        print("  pydub not installed; skipping post-processing.")
    except Exception as e:
        print(f"  WARNING: Post-processing failed ({e}); using raw TTS audio.")


# ---------------------------------------------------------------------------
# Phase 4 – Video Generation (Fal.ai)
# ---------------------------------------------------------------------------

def _upload_file(file_path):
    """Upload a local file to Fal storage and return the URL."""
    return fal_client.upload_file(file_path)


def _download_file(url, dest_path):
    """Download a file from a URL to a local path."""
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)


# ---- Text removal from actor image -----------------------------------------

def remove_text_from_image(image_path, output_dir, cost_tracker=None):
    """Detect and remove text/captions from an image using OCR + inpainting.

    Pipeline:
      1. Florence-2 OCR  -> bounding boxes of all text regions
      2. PIL              -> white-on-black mask from those boxes
      3. FLUX Pro Fill    -> inpaint the masked regions seamlessly
    """
    clean_path = os.path.join(output_dir, "actor_clean.jpg")
    if os.path.exists(clean_path):
        print("  Clean actor image already exists, skipping text removal.")
        return clean_path

    print("  Detecting text regions (Florence-2 OCR)...")
    image_url = _upload_file(image_path)

    ocr_result = fal_client.subscribe(
        "fal-ai/florence-2-large/ocr",
        arguments={"image_url": image_url},
    )
    if cost_tracker:
        cost_tracker.record_fal("fal-ai/florence-2-large/ocr", unit_count=1)

    # Florence-2 OCR returns results under varying keys; handle both formats
    results = ocr_result.get("results") or ocr_result.get("output") or {}
    boxes = results.get("quad_boxes", []) or results.get("bboxes", [])

    if not boxes:
        print("  No text detected in image; using original.")
        return image_path

    print(f"  Found {len(boxes)} text region(s). Building mask...")
    img = Image.open(image_path)
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)

    padding = 20  # expand mask beyond detected text edges
    for box in boxes:
        if isinstance(box, dict):
            x, y = box.get("x", 0), box.get("y", 0)
            w, h = box.get("w", 0), box.get("h", 0)
            draw.rectangle(
                [x - padding, y - padding, x + w + padding, y + h + padding],
                fill=255,
            )
        elif isinstance(box, (list, tuple)) and len(box) >= 4:
            # [x1, y1, x2, y2] format
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            draw.rectangle(
                [x1 - padding, y1 - padding, x2 + padding, y2 + padding],
                fill=255,
            )

    mask_path = os.path.join(output_dir, "text_mask.png")
    mask.save(mask_path)
    print(f"  Saved mask: {mask_path}")

    print("  Inpainting text regions (FLUX Pro Fill)...")
    mask_url = _upload_file(mask_path)

    inpaint_result = fal_client.subscribe(
        "fal-ai/flux-pro/v1/fill",
        arguments={
            "prompt": "clean natural background continuation, person in light blue shirt, no text, no captions, no watermark, seamless",
            "image_url": image_url,
            "mask_url": mask_url,
        },
    )
    if cost_tracker:
        cost_tracker.record_fal("fal-ai/flux-pro/v1/fill", unit_count=1)

    clean_url = inpaint_result["images"][0]["url"]
    _download_file(clean_url, clean_path)

    print(f"  -> Clean actor image saved: {clean_path}")
    return clean_path


# ---- Mode A: OmniHuman v1.5 (image + audio -> full-body talking video) --

def generate_omnihuman_video(actor_image, new_audio_path, output_dir,
                             resolution="720p", turbo=False, cost_tracker=None):
    """Generate a talking-head video using ByteDance OmniHuman v1.5.

    Takes a single image of the new actor and the audio track, produces a
    high-quality video with full-body motion, gestures, and lip sync.

    Limits: 30s audio at 1080p, 60s at 720p.
    """
    print("\n=== Phase 4: Video Generation (OmniHuman v1.5) ===")

    clip_path = os.path.join(output_dir, "omnihuman_video.mp4")
    if os.path.exists(clip_path):
        print(f"OmniHuman video already exists: {clip_path}")
        return [clip_path]

    print(f"  Actor image: {actor_image}")
    print(f"  Audio file:  {new_audio_path}")
    print(f"  Resolution:  {resolution}")

    # Check audio duration against limits
    audio_clip = AudioFileClip(new_audio_path)
    audio_dur = audio_clip.duration
    audio_clip.close()
    max_dur = 30 if resolution == "1080p" else 60
    if audio_dur > max_dur:
        print(f"  WARNING: Audio is {audio_dur:.1f}s but {resolution} limit is {max_dur}s.")
        print(f"  Audio will be trimmed to {max_dur}s.")

    print("  Uploading actor image...")
    image_url = _upload_file(actor_image)

    print("  Uploading audio...")
    audio_url = _upload_file(new_audio_path)

    print("  Generating video via OmniHuman v1.5...")
    print("  (this may take several minutes)")

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(f"    [OmniHuman] {log['message']}")

    try:
        args = {
            "image_url": image_url,
            "audio_url": audio_url,
            "resolution": resolution,
        }
        if turbo:
            args["turbo_mode"] = True

        result = fal_client.subscribe(
            "fal-ai/bytedance/omnihuman/v1.5",
            arguments=args,
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        video_url = result["video"]["url"]
    except Exception as e:
        print(f"  ERROR: OmniHuman v1.5 failed: {e}")
        sys.exit(1)

    print("  Downloading generated video...")
    _download_file(video_url, clip_path)

    if cost_tracker:
        cost_tracker.record_fal_video("fal-ai/bytedance/omnihuman/v1.5", audio_dur)

    size_mb = os.path.getsize(clip_path) / (1024 * 1024)
    print(f"  -> Saved video: {clip_path} ({size_mb:.1f} MB)")
    return [clip_path]


# ---- Mode A-legacy: SadTalker (free, lower quality) ---------------------

def generate_sadtalker_video(actor_image, new_audio_path, output_dir, cost_tracker=None):
    """Generate a talking-head video using SadTalker (free, face-only)."""
    print("\n=== Phase 4: Video Generation (SadTalker) ===")

    clip_path = os.path.join(output_dir, "sadtalker_video.mp4")
    if os.path.exists(clip_path):
        print(f"SadTalker video already exists: {clip_path}")
        return [clip_path]

    print(f"  Actor image: {actor_image}")
    print(f"  Audio file:  {new_audio_path}")

    print("  Uploading actor image...")
    image_url = _upload_file(actor_image)

    print("  Uploading audio...")
    audio_url = _upload_file(new_audio_path)

    print("  Generating video via SadTalker...")

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(f"    [SadTalker] {log['message']}")

    try:
        result = fal_client.subscribe(
            "fal-ai/sadtalker",
            arguments={
                "source_image_url": image_url,
                "driven_audio_url": audio_url,
                "face_model_resolution": "512",
                "expression_scale": 1.0,
                "still_mode": False,
                "preprocess": "full",
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        video_url = result["video"]["url"]
    except Exception as e:
        print(f"  ERROR: SadTalker failed: {e}")
        sys.exit(1)

    print("  Downloading generated video...")
    _download_file(video_url, clip_path)

    if cost_tracker:
        cost_tracker.record_fal("fal-ai/sadtalker", unit_count=1)

    size_mb = os.path.getsize(clip_path) / (1024 * 1024)
    print(f"  -> Saved video: {clip_path} ({size_mb:.1f} MB)")
    return [clip_path]


# ---- Mode B: Scene-based generation (Luma Dream Machine) ----------------

def generate_scene_videos(keyframe_dir, transcript, output_dir, prompt_template=None, cost_tracker=None):
    """Generate a new video clip for each scene keyframe via Luma Dream Machine."""
    print("\n=== Phase 4: Scene Video Generation (Luma) ===")

    clips_dir = os.path.join(output_dir, "generated_clips")
    os.makedirs(clips_dir, exist_ok=True)

    keyframe_files = sorted(
        glob.glob(os.path.join(keyframe_dir, "*.jpg")),
        key=lambda p: os.path.basename(p),
    )

    if not keyframe_files:
        print("ERROR: No keyframes found. Cannot generate videos.")
        sys.exit(1)

    # Build per-scene text snippets from transcript segments
    segments = transcript.get("segments", [])
    num_scenes = len(keyframe_files)
    scene_texts = [""] * num_scenes
    if segments:
        total_duration = segments[-1]["end"] if segments else 0
        for seg in segments:
            mid = (seg["start"] + seg["end"]) / 2
            scene_idx = min(int(mid / total_duration * num_scenes), num_scenes - 1)
            scene_texts[scene_idx] += " " + seg["text"]
    scene_texts = [t.strip() for t in scene_texts]

    default_template = (
        "A cinematic shot matching this scene. "
        "Maintain the same visual style and setting as the reference image. "
        "Scene context: {scene_text}"
    )
    template = prompt_template or default_template

    generated_paths = []
    for i, kf_path in enumerate(keyframe_files):
        scene_num = i + 1
        clip_path = os.path.join(clips_dir, f"scene_{scene_num:03d}.mp4")

        if os.path.exists(clip_path):
            print(f"  Scene {scene_num}/{num_scenes}: already generated, skipping.")
            generated_paths.append(clip_path)
            continue

        scene_text = scene_texts[i] if scene_texts[i] else "A person speaking to camera."
        prompt = template.format(scene_text=scene_text)

        print(f"  Scene {scene_num}/{num_scenes}: uploading keyframe...")
        image_url = _upload_file(kf_path)

        print(f"  Scene {scene_num}/{num_scenes}: generating video...")
        try:
            result = fal_client.subscribe(
                "fal-ai/luma-dream-machine",
                arguments={
                    "prompt": prompt,
                    "image_url": image_url,
                    "aspect_ratio": "16:9",
                    "expand_prompt": True,
                },
            )
            video_url = result["video"]["url"]
        except Exception as e:
            print(f"  WARNING: Fal.ai generation failed for scene {scene_num}: {e}")
            print("  Skipping this scene.")
            continue

        print(f"  Scene {scene_num}/{num_scenes}: downloading clip...")
        _download_file(video_url, clip_path)

        if cost_tracker:
            cost_tracker.record_fal("fal-ai/luma-dream-machine", unit_count=1)

        generated_paths.append(clip_path)
        print(f"  -> Saved {clip_path}")

    print(f"  Total clips generated: {len(generated_paths)}")
    return generated_paths


# ---------------------------------------------------------------------------
# Phase 5 – Final Assembly
# ---------------------------------------------------------------------------

def assemble_final_video(video_clip_paths, new_audio_path, output_dir):
    """Concatenate generated clips and overlay the new audio track."""
    print("\n=== Phase 5: Final Assembly ===")

    if not video_clip_paths:
        print("ERROR: No video clips to assemble.")
        sys.exit(1)

    final_path = os.path.join(output_dir, "final_video.mp4")

    print("Loading clips...")
    clips = [VideoFileClip(p) for p in video_clip_paths]
    combined = concatenate_videoclips(clips, method="compose")

    print("Attaching new audio...")
    new_audio = AudioFileClip(new_audio_path)

    # Match durations: trim audio or video to the shorter one
    if new_audio.duration > combined.duration:
        new_audio = new_audio.subclipped(0, combined.duration)
    elif combined.duration > new_audio.duration:
        combined = combined.subclipped(0, new_audio.duration)

    combined = combined.with_audio(new_audio)

    print("Writing final video...")
    combined.write_videofile(
        final_path,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )

    # Cleanup
    for c in clips:
        c.close()
    combined.close()
    new_audio.close()

    print(f"\n{'='*50}")
    print(f"  Done! Final video saved to: {final_path}")
    print(f"{'='*50}")
    return final_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Powered Video Recreation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use an existing actor image
  python recreate_video.py Videos/Lujo_-_José.mp4 --actor-image actor.jpg

  # Generate a new actor with AI (analyzes original video for scene matching)
  python recreate_video.py Videos/Lujo_-_José.mp4 --generate-actor "A blonde woman in her 30s"

  # Scene-based generation (Luma Dream Machine, no actor needed)
  python recreate_video.py Videos/Lujo_-_José.mp4 --mode scenes
        """,
    )
    parser.add_argument("video", help="Path to the source video file")
    parser.add_argument(
        "--actor-image",
        default=None,
        help="Path to an existing actor image file.",
    )
    parser.add_argument(
        "--generate-actor",
        default=None,
        metavar="DESCRIPTION",
        help="Generate a new actor with AI. Describe the person's appearance "
             "(e.g. 'A blonde woman in her 30s'). The scene, lighting, and "
             "framing are automatically matched to the original video.",
    )
    parser.add_argument(
        "--voice-id",
        default=None,
        help="ElevenLabs voice ID (default: from VOICE_ID env var)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--mode",
        default="omnihuman",
        choices=["omnihuman", "sadtalker", "scenes"],
        help="Video generation mode: 'omnihuman' (OmniHuman v1.5, best quality), "
             "'sadtalker' (free, face-only), 'scenes' (per-scene Luma). Default: omnihuman",
    )
    parser.add_argument(
        "--resolution",
        default="720p",
        choices=["720p", "1080p"],
        help="Video resolution for OmniHuman mode (default: 720p). "
             "720p supports up to 60s audio, 1080p up to 30s.",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Enable turbo mode for OmniHuman (faster but may reduce quality)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom prompt template for scene mode. Use {scene_text} as placeholder.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <video_name>_output)",
    )
    parser.add_argument(
        "--no-clean-image",
        action="store_true",
        help="Skip automatic text/caption removal from actor image",
    )
    parser.add_argument(
        "--skip-video-gen",
        action="store_true",
        help="Skip the video generation phase (useful for testing audio only)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    # Output directory – auto-version: <name>_v1, _v2, _v3 …
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        parent = os.path.dirname(video_path) or "."
        version = 1
        while True:
            output_dir = os.path.join(parent, f"{base_name}_v{version}")
            if not os.path.exists(output_dir):
                break
            version += 1
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Cost tracking (saved to costs.json in project root and output_dir)
    project_root = os.path.dirname(os.path.abspath(__file__))
    cost_tracker = CostTracker(project_root=project_root)
    cost_tracker.set_run_meta(video_path, output_dir, args.mode)

    # Phase 1: Analyze
    audio_path, keyframe_dir = analyze_video(video_path, output_dir)

    # Phase 2: Transcribe
    transcript = transcribe_audio(audio_path, output_dir, args.whisper_model)

    # Phase 3: Recreate audio
    voice_id = args.voice_id or DEFAULT_VOICE_ID
    new_audio_path = recreate_audio(transcript, voice_id, output_dir, cost_tracker=cost_tracker)

    if args.skip_video_gen:
        print("\n--skip-video-gen set. Stopping before video generation.")
        print(f"Transcript: {os.path.join(output_dir, 'transcript.json')}")
        print(f"New audio:  {new_audio_path}")
        run_path, global_path = cost_tracker.save_run(output_dir)
        print(f"Costs saved: {run_path} | {global_path}")
        return

    # Phase 4: Generate videos
    if args.mode in ("omnihuman", "sadtalker"):
        # Resolve actor image: --generate-actor > --actor-image > first keyframe
        keyframes = sorted(glob.glob(os.path.join(keyframe_dir, "*.jpg")))
        if args.generate_actor:
            ref_keyframe = keyframes[0] if keyframes else None
            if not ref_keyframe:
                print("ERROR: No keyframes found to analyze scene for actor generation.")
                sys.exit(1)
            actor_image = generate_actor_image(
                args.generate_actor, ref_keyframe, output_dir
            )
        elif args.actor_image:
            actor_image = args.actor_image
        elif keyframes:
            actor_image = keyframes[0]
            print(f"No --actor-image provided; using first keyframe: {actor_image}")
        else:
            print("ERROR: No --actor-image, --generate-actor, and no keyframes found.")
            sys.exit(1)

        if not args.generate_actor and actor_image and not os.path.exists(actor_image):
            print(f"ERROR: Actor image not found: {actor_image}")
            sys.exit(1)

        # Remove text/captions from actor image unless disabled
        if not args.no_clean_image:
            print("\n=== Cleaning actor image (removing text/captions) ===")
            actor_image = remove_text_from_image(actor_image, output_dir, cost_tracker=cost_tracker)

        if args.mode == "omnihuman":
            generated_clips = generate_omnihuman_video(
                actor_image, new_audio_path, output_dir,
                resolution=args.resolution, turbo=args.turbo,
                cost_tracker=cost_tracker,
            )
        else:
            generated_clips = generate_sadtalker_video(
                actor_image, new_audio_path, output_dir,
                cost_tracker=cost_tracker,
            )
    else:
        generated_clips = generate_scene_videos(
            keyframe_dir, transcript, output_dir, args.prompt,
            cost_tracker=cost_tracker,
        )

    # Phase 5: Assemble
    assemble_final_video(generated_clips, new_audio_path, output_dir)

    # Save cost tracking
    run_path, global_path = cost_tracker.save_run(output_dir)
    print(f"\nCosts saved: {run_path} | {global_path}")
    print(f"  Est. total: ${cost_tracker._run_total_usd():.4f} USD")


if __name__ == "__main__":
    main()
