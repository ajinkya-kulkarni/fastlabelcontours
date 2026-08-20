from __future__ import annotations

import numpy as np
import pytest

from fastlabelcontours import LabelContours, contours


def _contour_points(result: LabelContours, i: int) -> np.ndarray:
    return result.points[result.contour_offsets[i] : result.contour_offsets[i + 1]]


def _signed_area(points: np.ndarray) -> float:
    x = points[:, 0].astype(np.int64)
    y = points[:, 1].astype(np.int64)
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0


def _areas_by_label(labels: np.ndarray) -> dict[int, float]:
    result = contours(labels)
    out: dict[int, float] = {}
    for label_index, label in enumerate(result.ids):
        first = int(result.label_offsets[label_index])
        last = int(result.label_offsets[label_index + 1])
        out[int(label)] = sum(_signed_area(_contour_points(result, i)) for i in range(first, last))
    return out


def test_single_pixel_exact_edges() -> None:
    labels = np.array([[7]], dtype=np.uint32)
    result = contours(labels)

    np.testing.assert_array_equal(result.ids, np.array([7], dtype=np.uint32))
    np.testing.assert_array_equal(result.contour_offsets, np.array([0, 4], dtype=np.intp))
    np.testing.assert_array_equal(result.label_offsets, np.array([0, 1], dtype=np.intp))
    np.testing.assert_array_equal(result.is_hole, np.array([False]))
    np.testing.assert_array_equal(
        result.points,
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int32),
    )


def test_hole_is_preserved_and_area_is_exact() -> None:
    labels = np.ones((5, 5), dtype=np.uint32)
    labels[1:4, 1:4] = 0
    result = contours(labels)

    assert result.ids.tolist() == [1]
    assert result.is_hole.tolist().count(False) == 1
    assert result.is_hole.tolist().count(True) == 1
    areas = [_signed_area(_contour_points(result, i)) for i in range(2)]
    assert sorted(areas) == [-9.0, 25.0]
    assert sum(areas) == 16.0


def test_disconnected_same_label_has_multiple_outer_contours() -> None:
    labels = np.array([[5, 0, 5]], dtype=np.uint32)
    result = contours(labels)

    assert result.ids.tolist() == [5]
    assert result.label_offsets.tolist() == [0, 2]
    assert result.is_hole.tolist() == [False, False]
    assert [_signed_area(_contour_points(result, i)) for i in range(2)] == [1.0, 1.0]


def test_diagonal_pixels_are_separate_four_connected_components() -> None:
    labels = np.array([[3, 0], [0, 3]], dtype=np.uint32)
    result = contours(labels)

    assert result.label_offsets.tolist() == [0, 2]
    assert result.is_hole.tolist() == [False, False]
    assert [_signed_area(_contour_points(result, i)) for i in range(2)] == [1.0, 1.0]


def test_touching_different_labels_get_independent_boundaries() -> None:
    labels = np.array([[1, 2]], dtype=np.uint32)
    result = contours(labels)

    assert result.ids.tolist() == [1, 2]
    assert result.label_offsets.tolist() == [0, 1, 2]
    assert _areas_by_label(labels) == {1: 1.0, 2: 1.0}


def test_sparse_uint64_labels_and_exact_area() -> None:
    labels = np.array(
        [
            [0, 2**40, 2**40, 0],
            [9, 9, 0, 0],
            [9, 0, 17, 17],
        ],
        dtype=np.uint64,
    )
    result = contours(labels)

    assert result.ids.dtype == np.uint64
    assert result.ids.tolist() == [9, 17, 2**40]
    assert _areas_by_label(labels) == {9: 3.0, 17: 2.0, 2**40: 2.0}


def test_noncontiguous_input_is_supported() -> None:
    base = np.array([[1, 1, 0, 0], [0, 2, 2, 0]], dtype=np.uint32)
    labels = base[:, ::-1]
    assert not labels.flags.c_contiguous
    assert _areas_by_label(labels) == {1: 2.0, 2: 2.0}


def test_empty_and_all_background() -> None:
    for labels in (np.zeros((0, 0), dtype=np.uint32), np.zeros((3, 4), dtype=np.uint32)):
        result = contours(labels)
        assert result.ids.size == 0
        assert result.points.shape == (0, 2)
        assert result.contour_offsets.tolist() == [0]
        assert result.label_offsets.tolist() == [0]
        assert result.is_hole.size == 0


def test_validation() -> None:
    with pytest.raises(TypeError):
        contours(np.zeros((2, 2), dtype=np.int32))
    with pytest.raises(ValueError):
        contours(np.zeros((2, 2, 1), dtype=np.uint32))
    with pytest.raises(TypeError):
        contours([[1, 2]])  # type: ignore[arg-type]


def _returned_edges(labels: np.ndarray) -> set[tuple[int, int, int, int, int]]:
    result = contours(labels)
    edges: set[tuple[int, int, int, int, int]] = set()
    for label_index, label in enumerate(result.ids):
        first = int(result.label_offsets[label_index])
        last = int(result.label_offsets[label_index + 1])
        for contour_index in range(first, last):
            points = _contour_points(result, contour_index)
            for i in range(len(points)):
                x0, y0 = (int(v) for v in points[i])
                x1, y1 = (int(v) for v in points[(i + 1) % len(points)])
                edges.add((int(label), x0, y0, x1, y1))
    return edges


def _expected_edges(labels: np.ndarray) -> set[tuple[int, int, int, int, int]]:
    edges: set[tuple[int, int, int, int, int]] = set()
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label == 0:
                continue
            if y == 0 or labels[y - 1, x] != label:
                edges.add((label, x, y, x + 1, y))
            if x + 1 == width or labels[y, x + 1] != label:
                edges.add((label, x + 1, y, x + 1, y + 1))
            if y + 1 == height or labels[y + 1, x] != label:
                edges.add((label, x + 1, y + 1, x, y + 1))
            if x == 0 or labels[y, x - 1] != label:
                edges.add((label, x, y + 1, x, y))
    return edges


def test_randomized_boundaries_match_label_image_exactly() -> None:
    rng = np.random.default_rng(1234)
    for _ in range(50):
        labels = rng.integers(0, 6, size=(8, 9), dtype=np.uint32)
        assert _returned_edges(labels) == _expected_edges(labels)
