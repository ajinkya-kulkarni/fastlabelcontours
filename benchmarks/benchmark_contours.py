from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from rasterio.features import shapes
from scipy import ndimage

from fastlabelcontours import contours


def make_blob_labels(size: int, instances: int, seed: int = 0) -> np.ndarray:
    """Create separated ellipse-like instances without full-image temporaries."""
    labels = np.zeros((size, size), dtype=np.uint32)
    side = int(np.ceil(np.sqrt(instances)))
    cell = max(size // side, 1)
    rng = np.random.default_rng(seed)

    label = 1
    for gy in range(side):
        for gx in range(side):
            if label > instances:
                return labels
            y0 = gy * cell
            x0 = gx * cell
            y1 = min((gy + 1) * cell, size)
            x1 = min((gx + 1) * cell, size)
            h = y1 - y0
            w = x1 - x0
            if h < 3 or w < 3:
                label += 1
                continue

            cy = (h - 1) / 2.0 + rng.uniform(-0.08, 0.08) * h
            cx = (w - 1) / 2.0 + rng.uniform(-0.08, 0.08) * w
            ry = max(1.0, rng.uniform(0.25, 0.42) * h)
            rx = max(1.0, rng.uniform(0.25, 0.42) * w)
            yy, xx = np.ogrid[:h, :w]
            mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
            labels[y0:y1, x0:x1][mask] = label
            label += 1
    return labels


def opencv_full_image(labels: np.ndarray) -> int:
    n_contours = 0
    for label in np.unique(labels):
        if label == 0:
            continue
        binary = (labels == label).astype(np.uint8)
        found, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        n_contours += len(found)
    return n_contours


def opencv_cropped(labels: np.ndarray) -> int:
    n_contours = 0
    for label, bbox in enumerate(ndimage.find_objects(labels), start=1):
        if bbox is None:
            continue
        binary = (labels[bbox] == label).astype(np.uint8)
        found, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        n_contours += len(found)
    return n_contours


def rasterio_shapes(labels: np.ndarray) -> int:
    return sum(1 for _geometry, _value in shapes(labels, mask=labels != 0, connectivity=4))


def median_ms(
    fn: Callable[[np.ndarray], Any], labels: np.ndarray, repeat: int
) -> tuple[float, Any]:
    times: list[float] = []
    out: Any = None
    for _ in range(repeat):
        start = time.perf_counter()
        out = fn(labels)
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times), out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--instances", type=int, default=4096)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    labels = make_blob_labels(args.size, args.instances)
    contours(labels)  # warm-up

    methods: list[tuple[str, Callable[[np.ndarray], Any]]] = [
        ("fastlabelcontours", contours),
        ("OpenCV, bbox-cropped", opencv_cropped),
        ("Rasterio shapes", rasterio_shapes),
        ("OpenCV, full-image per label", opencv_full_image),
    ]

    print(f"shape: {labels.shape}, instances: {args.instances}")
    for name, fn in methods:
        ms, out = median_ms(fn, labels, args.repeat)
        count = len(out.is_hole) if name == "fastlabelcontours" else out
        print(f"{name:29s} {ms:9.2f} ms  ({count} contours)")


if __name__ == "__main__":
    main()
