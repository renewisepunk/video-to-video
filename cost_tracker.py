"""
Cost tracking for API usage (ElevenLabs, Fal.ai).

Saves usage and estimated costs locally:
  - Project root: costs.json (cumulative across all runs)
  - Per-run: <output_dir>/costs.json (costs for that run only)
"""

import os
import json
from datetime import datetime
from typing import Any

# Default pricing (USD) – update as needed; Fal.ai pricing varies by account
DEFAULT_PRICING = {
    "elevenlabs": {
        "chars_per_usd": 8_333,  # ~$0.12 per 1K chars for eleven_v3
    },
    "fal": {
        "fal-ai/florence-2-large/ocr": {"unit": "image", "usd_per_unit": 0.01},
        "fal-ai/flux-pro/v1/fill": {"unit": "image", "usd_per_unit": 0.04},
        "fal-ai/bytedance/omnihuman/v1.5": {"unit": "second", "usd_per_unit": 0.16},
        "fal-ai/sadtalker": {"unit": "video", "usd_per_unit": 0.05},
        "fal-ai/luma-dream-machine": {"unit": "video", "usd_per_unit": 0.80},
    },
}


class CostTracker:
    """Tracks API usage and estimated costs, persisting to local JSON files."""

    def __init__(self, project_root: str | None = None):
        self.project_root = project_root or os.getcwd()
        self.global_path = os.path.join(self.project_root, "costs.json")
        self._run_entries: list[dict[str, Any]] = []
        self._run_meta: dict[str, Any] = {}

    def set_run_meta(self, video_path: str, output_dir: str, mode: str):
        """Set metadata for the current run."""
        self._run_meta = {
            "video": video_path,
            "output_dir": output_dir,
            "mode": mode,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def record_elevenlabs(self, characters: int, model_id: str = "eleven_v3"):
        """Record ElevenLabs TTS usage (character count)."""
        cfg = DEFAULT_PRICING["elevenlabs"]
        usd = characters / cfg["chars_per_usd"]
        self._run_entries.append({
            "service": "elevenlabs",
            "type": "tts",
            "model": model_id,
            "characters": characters,
            "est_cost_usd": round(usd, 4),
        })

    def record_fal(
        self,
        endpoint_id: str,
        unit_count: float = 1.0,
        unit: str | None = None,
    ):
        """Record Fal.ai API usage."""
        cfg = DEFAULT_PRICING["fal"].get(endpoint_id)
        if cfg:
            usd_per = cfg["usd_per_unit"]
            u = cfg.get("unit", "call")
        else:
            usd_per = 0.10  # fallback
            u = unit or "call"
        usd = unit_count * usd_per
        self._run_entries.append({
            "service": "fal",
            "endpoint": endpoint_id,
            "unit": u,
            "unit_count": unit_count,
            "est_cost_usd": round(usd, 4),
        })

    def record_fal_video(self, endpoint_id: str, duration_seconds: float):
        """Record Fal.ai video generation (billed per second)."""
        self.record_fal(endpoint_id, unit_count=duration_seconds, unit="second")

    def _run_total_usd(self) -> float:
        return sum(e.get("est_cost_usd", 0) for e in self._run_entries)

    def save_run(self, output_dir: str):
        """Save costs for this run to output_dir and append to global log."""
        run_data = {
            **self._run_meta,
            "entries": self._run_entries.copy(),
            "total_est_usd": round(self._run_total_usd(), 4),
        }

        # Per-run file
        run_path = os.path.join(output_dir, "costs.json")
        os.makedirs(os.path.dirname(run_path) or ".", exist_ok=True)
        with open(run_path, "w") as f:
            json.dump(run_data, f, indent=2)

        # Append to global log
        global_data: dict[str, Any] = {"runs": [], "totals": {}}
        if os.path.exists(self.global_path):
            try:
                with open(self.global_path) as f:
                    global_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        global_data.setdefault("runs", []).append(run_data)

        # Recompute totals
        runs = global_data["runs"]
        total_usd = sum(r.get("total_est_usd", 0) for r in runs)
        eleven_chars = sum(
            e.get("characters", 0)
            for r in runs
            for e in r.get("entries", [])
            if e.get("service") == "elevenlabs"
        )
        fal_by_endpoint: dict[str, float] = {}
        for r in runs:
            for e in r.get("entries", []):
                if e.get("service") == "fal":
                    ep = e.get("endpoint", "unknown")
                    fal_by_endpoint[ep] = fal_by_endpoint.get(ep, 0) + e.get("est_cost_usd", 0)

        global_data["totals"] = {
            "total_est_usd": round(total_usd, 4),
            "elevenlabs_characters": eleven_chars,
            "fal_by_endpoint": fal_by_endpoint,
        }

        with open(self.global_path, "w") as f:
            json.dump(global_data, f, indent=2)

        return run_path, self.global_path

    def reset_run(self):
        """Clear entries for next run (e.g. when reusing tracker)."""
        self._run_entries.clear()
        self._run_meta.clear()
