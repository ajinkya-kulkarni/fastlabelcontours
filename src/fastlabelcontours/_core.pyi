from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

def contours2d(
    labels: NDArray[np.uint32] | NDArray[np.uint64],
) -> tuple[
    NDArray[np.uint32] | NDArray[np.uint64],
    NDArray[np.int32],
    NDArray[np.intp],
    NDArray[np.intp],
    NDArray[np.bool_],
]: ...
