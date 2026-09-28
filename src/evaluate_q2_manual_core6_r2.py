"""Evaluate frozen models against the corrected from-scratch R2 shorelines."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import evaluate_q2_manual_core6 as locked


PACK = locked.PACK
OUT = locked.ROOT / "preexperiment/results/q2_manual_core6_evaluation_R2_from_scratch_v2"


def r2_rows() -> list[dict]:
    source = PACK / "manual_review_R2_form.csv"
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 6:
        raise RuntimeError(f"Expected six R2 rows, found {len(rows)}")
    manifest = json.loads((PACK / "manual_review_manifest.json").read_text(encoding="utf-8"))
    metadata = {scene["scene_id"]: scene for scene in manifest["scenes"]}
    for row in rows:
        sid = row["scene_id"]
        row.update({key: metadata[sid][key] for key in
                    ("lake", "season", "l8_date", "s2_date")})
        path = PACK / sid / "shoreline_REVIEWED_R2.geojson"
        if not path.exists():
            raise FileNotFoundError(path)
        row.update({
            "reviewed_path": path,
            "consensus_status": "SINGLE_HUMAN_REVIEWER_FROM_SCRATCH_R2",
        })
    return rows


def main() -> None:
    locked.OUT = OUT
    locked.review_rows = r2_rows
    locked.main()


if __name__ == "__main__":
    main()
