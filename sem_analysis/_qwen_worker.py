"""Isolated Qwen worker — runs in a child process so crashes don't kill the API."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: _qwen_worker.py image.npy config.json out.json", file=sys.stderr)
        return 2
    img_path, cfg_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    img = np.load(img_path)
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    from sem_analysis.qwen_peaks import detect_apex_coordinates

    pts = detect_apex_coordinates(img, cfg)
    Path(out_path).write_text(json.dumps(pts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
