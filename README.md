# fastlabelcontours

Fast direct contour extraction from 2D integer label images.

`fastlabelcontours` traces every nonzero label directly from one integer label map.
It does not create a full binary mask for each instance.

```python
import numpy as np
from fastlabelcontours import contours

labels = np.array(
    [[0, 7, 7],
     [0, 7, 0],
     [9, 9, 0]],
    dtype=np.uint32,
)

result = contours(labels)
```

The result is a compact ragged representation:

- `ids`: observed nonzero labels, sorted ascending
- `points`: concatenated `(x, y)` pixel-edge vertices
- `contour_offsets`: slices `points` into individual contours
- `label_offsets`: slices contours into labels
- `is_hole`: marks inner boundaries

Contours are implicitly closed; the first point is not repeated. Coordinates follow
exact pixel edges, so signed contour area sums to the number of pixels carrying each
label. Diagonal-only contact does not merge components.

## Why

A common implementation repeatedly materializes `labels == label` and calls a contour
routine once per instance. Even a better implementation still has to discover each
instance bounding box, crop it, materialize a binary mask, and invoke the contour tracer
once per object.

`fastlabelcontours` instead scans the label image once and traces label boundaries
directly in C++.

On a local 2048 x 2048 synthetic blob mask with 4096 instances:

| Method | Median time | Relative to fastlabelcontours |
| --- | ---: | ---: |
| `fastlabelcontours` | ~25 ms | 1.0x |
| SciPy bbox discovery + cropped OpenCV | ~55 ms | ~2.2x slower |
| Rasterio `features.shapes` | ~105 ms | ~4.2x slower |
| full-image `labels == id` + OpenCV per instance | ~4.95 s | ~199x slower |

The bbox-cropped OpenCV result is the most useful baseline: it avoids rescanning the full
image for every label and still takes roughly twice as long in this workload. Exact
results are machine- and data-dependent; run the included benchmark on your own masks.

## Installation

```bash
pip install fastlabelcontours
```

Until wheels are published, install from source with a C++17 compiler:

```bash
pip install .
```

## API

```python
from fastlabelcontours import contours

result = contours(labels)

# contour i
points_i = result.points[result.contour_offsets[i] : result.contour_offsets[i + 1]]

# contours belonging to label j
first = result.label_offsets[j]
last = result.label_offsets[j + 1]
```

Inputs must be 2D NumPy arrays with dtype `uint32` or `uint64`. Label `0` is background.
The public API copies non-contiguous inputs to contiguous storage before entering the C++
core.

## Geometry semantics

- Coordinates are integer `(x, y)` pixel-edge vertices, not pixel centers.
- Contours are implicitly closed.
- Outer contours have positive signed area; holes have negative signed area.
- `is_hole` records the same distinction explicitly.
- A label may have multiple disconnected outer contours.
- Connectivity is 4-connected, so diagonal-only contact does not merge components.
- No simplification is performed; the output represents exact raster boundaries.

## Scope

`fastlabelcontours` deliberately stays narrow:

- 2D `uint32` / `uint64` label images
- background label `0`
- CPU only
- NumPy as the only runtime dependency
- exact contours, including holes and disconnected components
- no polygon simplification or general geometry operations

## Development

```bash
python -m pip install -e '.[dev,benchmark]'
pytest
ruff check .
mypy src/fastlabelcontours
python benchmarks/benchmark_contours.py --size 2048 --instances 4096
```

Build and validate release artifacts with:

```bash
python -m build
python -m twine check dist/*
```

`cibuildwheel` configuration for CPython 3.10-3.13 on Linux, macOS, and Windows is
included in `pyproject.toml` for producing binary wheels.

## License

MIT
