#!/usr/bin/env python3
"""Convert Whisper word-level transcription output to Remotion Caption[] JSON format."""

import json
import sys


def whisper_to_remotion_captions(whisper_result, output_path):
    """Convert Whisper word-level output to Remotion Caption[] format.

    Args:
        whisper_result: Whisper transcription dict with segments[].words[]
        output_path: Path to write the Remotion captions JSON

    Returns:
        List of Remotion Caption dicts
    """
    captions = []

    for segment in whisper_result.get("segments", []):
        words = segment.get("words", [])
        if not words:
            continue

        for word_info in words:
            text = word_info.get("word", "")
            start = word_info.get("start", 0)
            end = word_info.get("end", 0)
            probability = word_info.get("probability", None)

            captions.append({
                "text": text,
                "startMs": round(start * 1000),
                "endMs": round(end * 1000),
                "timestampMs": None,
                "confidence": round(probability, 4) if probability is not None else None,
            })

    with open(output_path, "w") as f:
        json.dump(captions, f, indent=2)

    return captions


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_captions.py <whisper_input.json> <remotion_output.json>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    with open(input_path, "r") as f:
        whisper_result = json.load(f)

    captions = whisper_to_remotion_captions(whisper_result, output_path)
    print(f"Converted {len(captions)} words -> {output_path}")
