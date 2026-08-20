from __future__ import annotations

from typing import NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

from . import _core

LabelArray: TypeAlias = NDArray[np.uint32] | NDArray[np.uint64]

_SUPPORTED_DTYPES = (np.dtype(np.uint32), np.dtype(np.uint64))


class LabelContours(NamedTuple):
    """Compact ragged contours grouped by label.

    ``points[contour_offsets[i]:contour_offsets[i + 1]]`` contains contour ``i``.
    ``label_offsets[j]:label_offsets[j + 1]`` gives the contour range for
    ``ids[j]``. Contours are implicitly closed; the first point is not repeated.
    ``is_hole`` is true for counter-wound inner boundaries.
    """

    ids: LabelArray
    points: NDArray[np.int32]
    contour_offsets: NDArray[np.intp]
    label_offsets: NDArray[np.intp]
    is_hole: NDArray[np.bool_]


def contours(labels: LabelArray) -> LabelContours:
    """Extract exact pixel-edge contours directly from a 2D integer label image.

    Background label ``0`` is ignored. The input must have dtype ``uint32`` or
    ``uint64``. All nonzero labels are processed in one raster scan without
    materializing one binary mask per instance.

    Coordinates are ``(x, y)`` integer pixel-edge vertices. Outer contours and
    holes have opposite winding. Pixels connected only diagonally are represented
    as separate 4-connected components.
    """
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a NumPy array")
    if labels.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("labels dtype must be uint32 or uint64")
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")

    ids, points, contour_offsets, label_offsets, is_hole = _core.contours2d(
        np.ascontiguousarray(labels)
    )
    return LabelContours(ids, points, contour_offsets, label_offsets, is_hole)
