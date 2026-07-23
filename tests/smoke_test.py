"""Offline smoke tests for the Forge adapter's non-model pieces.

Run from the extension root:
    python tests/smoke_test.py

This does not load SeedVR2 weights or require CUDA.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seedvr2_forge.config import SeedVR2Config
from seedvr2_forge.tiling import _axis_boxes, merge_tiles, split_image


def test_workflow_target() -> None:
    cfg = SeedVR2Config()
    assert cfg.target_long_edge() == 6080, cfg.target_long_edge()


def test_ttp_equivalent_6080_grid() -> None:
    boxes = _axis_boxes(6080, 1024)
    expected = [
        (0, 1024),
        (1012, 2036),
        (2024, 3048),
        (3036, 4060),
        (4048, 5072),
        (5056, 6080),
    ]
    assert boxes == expected, boxes


def test_split_merge_roundtrip() -> None:
    # Deliberately non-multiple dimensions to exercise adaptive overlap.
    h, w = 777, 1103
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack(
        [xx % 256, yy % 256, (xx // 2 + yy // 3) % 256], axis=-1
    ).astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    layout = split_image(image, tile_size=512, tile_padding=64)
    merged = merge_tiles(layout, [t.image for t in layout.tiles])
    out = np.asarray(merged, dtype=np.int16)
    src = np.asarray(image, dtype=np.int16)
    diff = np.abs(out - src)
    assert merged.size == image.size
    # Integer round-trip through floating crossfade should differ by at most 1-2 LSB.
    assert int(diff.max()) <= 2, int(diff.max())


if __name__ == "__main__":
    test_workflow_target()
    test_ttp_equivalent_6080_grid()
    test_split_merge_roundtrip()
    print("[PASS] SeedVR2 Forge offline smoke tests")
