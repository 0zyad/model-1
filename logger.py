"""
logger.py — Automatic result logging for Proppant QC.

Saves per-run results as:
  - JSON file (full details)
  - CSV row (appended to cumulative log)
  - Annotated overlay image (JPG)

All saved to ./logs/
"""
import json
import csv
from pathlib import Path
from datetime import datetime
import cv2
from config import LOG_DIR


class ResultLogger:
    """Logs analysis results to disk."""

    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "results.csv"
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "image_name",
                    "total_particles",
                    "pct_proppant_40_70",
                    "pct_proppant_20_40",
                    "verdict",
                    "reason",
                    "avg_confidence",
                    "blur_score",
                    "processing_time_sec",
                    "swe_all_passed",
                    "json_file",
                    "overlay_file",
                ])

    def log(self, result: dict) -> dict:
        """Log one analysis result. Returns dict with saved file paths."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(result.get("image_name", "unknown")).stem

        # ── JSON ──────────────────────────────────────────────────────
        json_name = f"{ts}_{stem}.json"
        json_path = self.log_dir / json_name
        json_data = {k: v for k, v in result.items() if k != "overlay"}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, default=str)

        # ── Overlay image ─────────────────────────────────────────────
        overlay_name = f"{ts}_{stem}_overlay.jpg"
        overlay_path = self.log_dir / overlay_name
        if result.get("overlay") is not None:
            cv2.imwrite(str(overlay_path), result["overlay"])

        # ── CSV (append) ──────────────────────────────────────────────
        comp = result.get("composition", {})
        swe = result.get("swe_checks", {})
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts,
                result.get("image_name", ""),
                result.get("total_particles", 0),
                comp.get("proppant_40_70", {}).get("percentage", 0),
                comp.get("proppant_20_40", {}).get("percentage", 0),
                result.get("verdict", ""),
                result.get("reason", ""),
                result.get("avg_confidence", 0),
                result.get("blur_score", 0),
                result.get("processing_time_sec", 0),
                swe.get("all_passed", False),
                json_name,
                overlay_name,
            ])

        return {
            "json_path": str(json_path),
            "overlay_path": str(overlay_path),
            "csv_path": str(self.csv_path),
        }
