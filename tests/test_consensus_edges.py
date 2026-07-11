"""Tests for multi-algorithm consensus edge detection."""

import numpy as np

from sem_analysis.consensus_edges import (
    build_consensus_edges,
    consensus_edge_map,
    edge_canny,
    edge_scharr,
    edge_sobel,
    run_edge_detectors,
)
from sem_analysis.pipeline import load_config


def _synthetic_blade(h=80, w=100) -> np.ndarray:
    """Bright V-shaped blade on dark background."""
    img = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2
    for y in range(10, h - 5):
        half = 2 + (y - 10) // 3
        x0 = max(0, cx - half)
        x1 = min(w, cx + half)
        img[y, x0:x1] = 200
    return img


class TestConsensusEdges:
    def test_vote_not_average(self):
        a = np.zeros((10, 10), dtype=np.uint8)
        b = np.zeros((10, 10), dtype=np.uint8)
        c = np.zeros((10, 10), dtype=np.uint8)
        a[5, 5] = 255
        b[5, 5] = 255
        # only 2/3 vote for (5,5); c empty
        consensus, meta = consensus_edge_map(
            {"a": a, "b": b, "c": c}, min_votes=2, vote_ratio=0.5
        )
        assert consensus[5, 5] == 255
        assert meta["strategy"] == "majority_vote_not_average"
        assert meta["votes_required"] == 2

    def test_single_vote_rejected_when_ratio_high(self):
        a = np.zeros((8, 8), dtype=np.uint8)
        b = np.zeros((8, 8), dtype=np.uint8)
        a[2, 2] = 255
        # ratio 1.0 → both methods must agree; lone vote is dropped
        consensus, meta = consensus_edge_map({"a": a, "b": b}, vote_ratio=1.0)
        assert meta["votes_required"] == 2
        assert consensus[2, 2] == 0

    def test_run_multiple_detectors(self):
        img = _synthetic_blade()
        cfg = load_config()
        maps = run_edge_detectors(
            img,
            ["canny", "sobel", "scharr", "laplacian", "adaptive_threshold", "otsu"],
            cfg,
        )
        assert "canny" in maps
        assert len(maps) >= 3

    def test_build_consensus_edges(self):
        img = _synthetic_blade().astype(np.float32) / 255.0
        cfg = load_config()
        edges, meta = build_consensus_edges(img, cfg)
        assert edges.shape == img.shape
        assert meta.get("strategy") == "majority_vote_not_average" or meta.get("fallback")
        assert np.count_nonzero(edges) > 0

    def test_canny_reference_thresholds(self):
        img = _synthetic_blade()
        edges = edge_canny(img, 40, 120)
        assert edges.dtype == np.uint8
        assert edges.max() in (0, 255)
