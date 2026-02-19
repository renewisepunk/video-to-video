#!/usr/bin/env python3
"""
Instagram-style animated caption overlay for video.

Accepts a video file and Whisper transcript segments (list of dicts with
'start', 'end', and 'text' keys) and burns in phrase-by-phrase bold white
captions with a black stroke, centered in the lower third.

Two rendering backends are provided:
  - MoviePy v2  (default): pure-Python, no ffmpeg binary knowledge required
  - FFmpeg drawtext (fallback / alternative): robust, handles complex unicode

Usage (standalone):
    python add_captions.py input.mp4 transcript.json output.mp4

Usage (as a module):
    from add_captions import add_captions_moviepy, add_captions_ffmpeg

    segments = [
        {"start": 0.0, "end": 2.5, "text": "Hello, welcome back"},
        {"start": 2.5, "end": 5.0, "text": "to my channel!"},
    ]
    add_captions_moviepy("input.mp4", segments, "output.mp4")
"""

import os
import re
import sys
import json
import shutil
import subprocess
import textwrap
import tempfile

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Maximum characters per caption line before wrapping.
# ~35 chars looks good at typical Reel/TikTok font sizes.
MAX_CHARS_PER_LINE = 35

# How many words to bundle into one caption phrase when the transcript
# segments are already at the sentence level (leave as-is) vs. word level.
# Set to 0 to use segments exactly as provided.
WORDS_PER_PHRASE = 0


def _clean_text(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def _wrap_text(text: str, max_chars: int = MAX_CHARS_PER_LINE) -> str:
    """Wrap a phrase to at most max_chars per line."""
    return "\n".join(textwrap.wrap(text, width=max_chars))


def _group_segments_into_phrases(segments, words_per_phrase: int = WORDS_PER_PHRASE):
    """
    Optionally re-group word-level Whisper segments into N-word phrases.

    If words_per_phrase == 0, the original segments are returned unchanged
    (useful when Whisper already returns sentence/phrase-level segments).

    Args:
        segments: list of {"start": float, "end": float, "text": str}
        words_per_phrase: number of words per output phrase (0 = no grouping)

    Returns:
        list of {"start": float, "end": float, "text": str}
    """
    if not words_per_phrase:
        return [
            {"start": s["start"], "end": s["end"], "text": _clean_text(s["text"])}
            for s in segments
        ]

    # Split every segment into individual words with proportional timestamps
    word_entries = []
    for seg in segments:
        words = _clean_text(seg["text"]).split()
        if not words:
            continue
        dur = (seg["end"] - seg["start"]) / len(words)
        for i, word in enumerate(words):
            word_entries.append({
                "start": seg["start"] + i * dur,
                "end": seg["start"] + (i + 1) * dur,
                "text": word,
            })

    # Re-bundle into fixed-size phrases
    phrases = []
    for i in range(0, len(word_entries), words_per_phrase):
        group = word_entries[i : i + words_per_phrase]
        phrases.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": " ".join(w["text"] for w in group),
        })

    return phrases


# ---------------------------------------------------------------------------
# Backend 1 – MoviePy v2
# ---------------------------------------------------------------------------

def add_captions_moviepy(
    input_video: str,
    segments: list,
    output_video: str,
    font_path: str = None,
    font_size: int = 70,
    color: str = "white",
    stroke_color: str = "black",
    stroke_width: int = 4,
    position_y_fraction: float = 0.72,
    words_per_phrase: int = WORDS_PER_PHRASE,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
) -> str:
    """
    Overlay Instagram-style animated captions using MoviePy v2.

    Captions appear phrase-by-phrase, bold, white, with a black stroke,
    centered horizontally and placed in the lower third of the frame.

    Args:
        input_video:         Path to the source video file.
        segments:            Whisper transcript segments – list of dicts with
                             keys 'start' (float), 'end' (float), 'text' (str).
        output_video:        Path where the output video will be written.
        font_path:           Absolute path to a .ttf/.otf bold font file.
                             Defaults to a system bold font (macOS/Linux).
        font_size:           Font size in points. 70 looks good for 1080p;
                             use 50-60 for 720p.
        color:               Text fill color (any CSS/Pillow color string).
        stroke_color:        Outline/border color.
        stroke_width:        Outline thickness in pixels (int). Must be an int
                             per MoviePy v2 API.
        position_y_fraction: Vertical position as a fraction of video height
                             (0.0 = top, 1.0 = bottom). 0.72 = lower third.
        words_per_phrase:    If > 0, re-group segments into N-word phrases.
        video_codec:         ffmpeg video codec for output.
        audio_codec:         ffmpeg audio codec for output.

    Returns:
        Absolute path to the written output video.
    """
    # ---- imports (MoviePy v2 – import from `moviepy`, NOT `moviepy.editor`) ----
    from moviepy import VideoFileClip, CompositeVideoClip
    from moviepy.video.VideoClip import TextClip  # v2 explicit path

    # ---- Resolve font --------------------------------------------------------
    if font_path is None:
        font_path = _find_bold_font()
    if not os.path.exists(font_path):
        raise FileNotFoundError(
            f"Font not found: {font_path}\n"
            "Pass font_path= explicitly or install a bold .ttf font."
        )

    # ---- Load video ----------------------------------------------------------
    video = VideoFileClip(input_video)
    video_w, video_h = video.size
    print(f"  Video: {video_w}x{video_h}, {video.duration:.2f}s")

    # ---- Build caption clips -------------------------------------------------
    phrases = _group_segments_into_phrases(segments, words_per_phrase)
    caption_clips = []

    for phrase in phrases:
        start = phrase["start"]
        end = phrase["end"]
        raw_text = phrase["text"]

        if not raw_text:
            continue

        # Wrap long phrases to multiple lines
        wrapped = _wrap_text(raw_text, max_chars=MAX_CHARS_PER_LINE)

        # Duration guard: ensure we have a positive, non-zero clip duration
        duration = max(end - start, 0.1)

        # ---- Create the text clip ----------------------------------------
        # MoviePy v2 TextClip constructor (from moviepy.video.VideoClip):
        #   TextClip(font, text, font_size, color, stroke_color, stroke_width,
        #            margin, method, text_align, horizontal_align,
        #            vertical_align, transparent, duration)
        #
        # IMPORTANT: stroke_width must be int (not float) in v2.
        # Add margin equal to stroke_width so the outline isn't clipped at
        # the bounding-box edge (known v2 issue fixed in master but may
        # still appear in release builds).
        margin_px = int(stroke_width) + 4  # a few extra pixels of safety

        txt_clip = TextClip(
            font=font_path,
            text=wrapped,
            font_size=font_size,
            color=color,
            stroke_color=stroke_color,
            stroke_width=int(stroke_width),
            margin=(margin_px, margin_px),
            method="label",          # "label" auto-sizes to text; use "caption"
                                     # with size=(video_w, None) for hard-wrap
            text_align="center",
            horizontal_align="center",
            vertical_align="center",
            transparent=True,
        )

        # ---- Position: horizontally centered, vertically in lower third ----
        # with_position accepts:
        #   ("center", y_pixels)        – center-x, absolute y
        #   (x_fraction, y_fraction)    – both fractions when relative=True
        #
        # We place the TOP of the text block at position_y_fraction * video_h,
        # then offset upward by half the clip height so the text is vertically
        # centered around that anchor point.
        txt_h = txt_clip.size[1]
        y_px = int(position_y_fraction * video_h - txt_h / 2)
        y_px = max(0, min(y_px, video_h - txt_h))  # clamp to frame

        txt_clip = (
            txt_clip
            .with_position(("center", y_px))
            .with_start(start)
            .with_duration(duration)
        )

        caption_clips.append(txt_clip)

    print(f"  Created {len(caption_clips)} caption clips.")

    if not caption_clips:
        print("  WARNING: No caption clips produced. Writing video without captions.")
        video.write_videofile(output_video, codec=video_codec, audio_codec=audio_codec)
        video.close()
        return os.path.abspath(output_video)

    # ---- Composite captions onto video ---------------------------------------
    final = CompositeVideoClip([video] + caption_clips)

    print(f"  Writing output to: {output_video}")
    final.write_videofile(
        output_video,
        codec=video_codec,
        audio_codec=audio_codec,
        fps=video.fps,
        logger="bar",
    )

    # ---- Cleanup -------------------------------------------------------------
    for c in caption_clips:
        c.close()
    final.close()
    video.close()

    return os.path.abspath(output_video)


# ---------------------------------------------------------------------------
# Backend 2 – FFmpeg drawtext (robust alternative / fallback)
# ---------------------------------------------------------------------------

def add_captions_ffmpeg(
    input_video: str,
    segments: list,
    output_video: str,
    font_path: str = None,
    font_size: int = 70,
    font_color: str = "white",
    border_color: str = "black",
    border_width: int = 4,
    position_y_fraction: float = 0.80,
    words_per_phrase: int = WORDS_PER_PHRASE,
    ffmpeg_bin: str = "ffmpeg",
) -> str:
    """
    Overlay Instagram-style animated captions using the FFmpeg drawtext filter.

    This is the most robust approach: it handles unicode well, doesn't need
    MoviePy's Pillow dependency, and produces identical output across platforms.
    Each caption phrase is added as a separate drawtext filter in a filter chain.

    Args:
        input_video:         Path to the source video file.
        segments:            Whisper transcript segments.
        output_video:        Path where the output video will be written.
        font_path:           Absolute path to a .ttf bold font file. If None,
                             a system font is located automatically.
        font_size:           Font size in pixels.
        font_color:          FFmpeg color string for text fill (e.g. "white").
        border_color:        FFmpeg color string for border (e.g. "black").
        border_width:        Border/outline thickness in pixels.
        position_y_fraction: Vertical anchor as fraction of video height.
                             The text is horizontally centered with
                             x=(w-text_w)/2 and vertically anchored at
                             y=h*fraction - text_h/2.
        words_per_phrase:    If > 0, re-group segments into N-word phrases.
        ffmpeg_bin:          Name or path of the ffmpeg executable.

    Returns:
        Absolute path to the written output video.

    Raises:
        RuntimeError: if ffmpeg is not found or exits with a non-zero code.
    """
    if not shutil.which(ffmpeg_bin):
        raise RuntimeError(
            f"ffmpeg not found at '{ffmpeg_bin}'. Install ffmpeg or pass ffmpeg_bin=."
        )

    if font_path is None:
        font_path = _find_bold_font()
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font not found: {font_path}")

    phrases = _group_segments_into_phrases(segments, words_per_phrase)

    # Build a chain of drawtext filters, one per caption phrase.
    # Each filter is enabled only during [start, end] seconds via the
    # `enable='between(t,start,end)'` option.
    filter_parts = []
    for phrase in phrases:
        start = phrase["start"]
        end = max(phrase["end"], start + 0.1)
        raw_text = phrase["text"]
        if not raw_text:
            continue

        # Wrap text (use \n for ffmpeg drawtext line breaks)
        wrapped = _wrap_text(raw_text, max_chars=MAX_CHARS_PER_LINE)

        # Escape special characters for drawtext:
        # single quotes, colons, backslashes are the usual suspects.
        escaped = (
            wrapped
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace(":", "\\:")
            .replace("\n", "\n")  # drawtext supports literal newlines in text=
        )

        # y position: center text block around position_y_fraction * h
        # (w-text_w)/2 centers horizontally; h*frac - text_h/2 for vertical
        y_expr = f"(h*{position_y_fraction}-text_h/2)"
        x_expr = "(w-text_w)/2"

        part = (
            f"drawtext="
            f"fontfile='{font_path}':"
            f"text='{escaped}':"
            f"fontsize={font_size}:"
            f"fontcolor={font_color}:"
            f"bordercolor={border_color}:"
            f"borderw={border_width}:"
            f"x={x_expr}:"
            f"y={y_expr}:"
            f"line_spacing=8:"
            f"enable='between(t,{start},{end})'"
        )
        filter_parts.append(part)

    if not filter_parts:
        print("  WARNING: No caption phrases to render; copying input to output.")
        shutil.copy2(input_video, output_video)
        return os.path.abspath(output_video)

    # Join all drawtext filters with commas (sequential application)
    vf = ",".join(filter_parts)

    cmd = [
        ffmpeg_bin,
        "-y",                    # overwrite output without asking
        "-i", input_video,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "copy",          # copy audio stream unchanged
        output_video,
    ]

    print(f"  Running ffmpeg with {len(filter_parts)} drawtext filters...")
    print(f"  Output: {output_video}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
        )

    return os.path.abspath(output_video)


# ---------------------------------------------------------------------------
# Font discovery helper
# ---------------------------------------------------------------------------

_CANDIDATE_FONTS = [
    # macOS – Supplemental folder (present on most macOS installs)
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    # macOS – other common locations
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    # Linux common locations
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    # Windows
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/impact.ttf",
]


def _find_bold_font() -> str:
    """
    Return the path to the first available bold font from a list of candidates.

    Raises FileNotFoundError if none are found.
    """
    for path in _CANDIDATE_FONTS:
        if os.path.exists(path):
            return path

    # Last resort: ask fontconfig (Linux/macOS with fc-list)
    try:
        result = subprocess.run(
            ["fc-list", "--format=%{file}\n", ":style=Bold"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            path = line.strip()
            if path and os.path.exists(path) and path.lower().endswith((".ttf", ".otf")):
                return path
    except Exception:
        pass

    raise FileNotFoundError(
        "No bold font found automatically.\n"
        "Pass font_path= explicitly, e.g.:\n"
        "  add_captions_moviepy(..., font_path='/path/to/Bold.ttf')"
    )


# ---------------------------------------------------------------------------
# Integration helper – wraps the full pipeline (transcribe + caption)
# ---------------------------------------------------------------------------

def add_captions_to_whisper_result(
    input_video: str,
    whisper_result: dict,
    output_video: str,
    backend: str = "moviepy",
    **kwargs,
) -> str:
    """
    High-level helper: takes the dict returned by whisper.transcribe() directly.

    Args:
        input_video:    Path to source video.
        whisper_result: The dict returned by ``whisper_model.transcribe(audio)``.
                        Must contain a 'segments' key.
        output_video:   Output path.
        backend:        'moviepy' (default) or 'ffmpeg'.
        **kwargs:       Forwarded to add_captions_moviepy / add_captions_ffmpeg.

    Returns:
        Absolute path to the output video.

    Example::

        import whisper
        from add_captions import add_captions_to_whisper_result

        model = whisper.load_model("base")
        result = model.transcribe("audio.mp3")

        add_captions_to_whisper_result(
            "input.mp4",
            result,
            "output_captioned.mp4",
            backend="moviepy",
            font_size=70,
        )
    """
    segments = whisper_result.get("segments", [])
    if not segments:
        raise ValueError("whisper_result contains no 'segments'. Run model.transcribe() first.")

    print(f"\n=== Adding Captions ({backend} backend) ===")
    print(f"  {len(segments)} transcript segments")

    if backend == "ffmpeg":
        return add_captions_ffmpeg(input_video, segments, output_video, **kwargs)
    else:
        return add_captions_moviepy(input_video, segments, output_video, **kwargs)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Add Instagram-style animated captions to a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MoviePy backend (default)
  python add_captions.py input.mp4 transcript.json output.mp4

  # FFmpeg backend
  python add_captions.py input.mp4 transcript.json output.mp4 --backend ffmpeg

  # Custom font and size
  python add_captions.py input.mp4 transcript.json output.mp4 \\
      --font /path/to/Bold.ttf --font-size 80

Transcript JSON format (Whisper output):
  {
    "segments": [
      {"start": 0.0, "end": 2.5, "text": "Hello everyone"},
      {"start": 2.5, "end": 5.0, "text": "welcome back"}
    ]
  }

Or just a bare list:
  [
    {"start": 0.0, "end": 2.5, "text": "Hello everyone"},
    ...
  ]
        """,
    )
    parser.add_argument("input_video", help="Path to input video")
    parser.add_argument("transcript_json", help="Path to Whisper transcript JSON")
    parser.add_argument("output_video", help="Path for output video")
    parser.add_argument(
        "--backend", default="moviepy", choices=["moviepy", "ffmpeg"],
        help="Rendering backend (default: moviepy)"
    )
    parser.add_argument("--font", default=None, help="Path to a bold .ttf/.otf font file")
    parser.add_argument("--font-size", type=int, default=70, help="Font size in points (default: 70)")
    parser.add_argument("--stroke-width", type=int, default=4, help="Outline thickness in pixels (default: 4)")
    parser.add_argument(
        "--position-y", type=float, default=0.80,
        help="Vertical position as fraction of frame height (default: 0.80)"
    )
    parser.add_argument(
        "--words-per-phrase", type=int, default=0,
        help="Re-group into N-word phrases (0 = use segments as-is, default: 0)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_video):
        print(f"ERROR: Input video not found: {args.input_video}")
        sys.exit(1)
    if not os.path.exists(args.transcript_json):
        print(f"ERROR: Transcript JSON not found: {args.transcript_json}")
        sys.exit(1)

    with open(args.transcript_json, "r") as f:
        data = json.load(f)

    # Accept either the full Whisper result dict or a bare list of segments
    if isinstance(data, list):
        segments = data
    elif isinstance(data, dict) and "segments" in data:
        segments = data["segments"]
    else:
        print("ERROR: JSON must be either a list of segments or a Whisper result dict.")
        sys.exit(1)

    shared_kwargs = dict(
        font_path=args.font,
        font_size=args.font_size,
        stroke_width=args.stroke_width,
        position_y_fraction=args.position_y,
        words_per_phrase=args.words_per_phrase,
    )

    if args.backend == "ffmpeg":
        shared_kwargs["border_width"] = shared_kwargs.pop("stroke_width")
        add_captions_ffmpeg(args.input_video, segments, args.output_video, **shared_kwargs)
    else:
        add_captions_moviepy(args.input_video, segments, args.output_video, **shared_kwargs)

    print(f"\nDone! Captioned video saved to: {args.output_video}")


if __name__ == "__main__":
    _cli()
