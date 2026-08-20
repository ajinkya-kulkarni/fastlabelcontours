import numpy as np

from fastlabelcontours import contours

labels = np.array(
    [
        [0, 7, 7, 0],
        [0, 7, 0, 0],
        [9, 9, 0, 9],
    ],
    dtype=np.uint32,
)

result = contours(labels)
print("ids:", result.ids)
print("points:\n", result.points)
print("contour_offsets:", result.contour_offsets)
print("label_offsets:", result.label_offsets)
print("is_hole:", result.is_hole)
