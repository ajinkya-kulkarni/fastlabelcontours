#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_20_API_VERSION
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Vertex {
    std::int32_t x;
    std::int32_t y;
};

struct BoundaryEdge {
    npy_intp pixel_index;
    std::uint8_t side;
    std::uint8_t dir;
    Vertex a;
    Vertex b;
};

struct Loop {
    std::uint64_t label;
    std::vector<Vertex> points;
    bool is_hole;
};

inline bool same_vertex(const Vertex& a, const Vertex& b) {
    return a.x == b.x && a.y == b.y;
}

inline BoundaryEdge edge_for_pixel(npy_intp y, npy_intp x, npy_intp width, std::uint8_t side) {
    const auto xi = static_cast<std::int32_t>(x);
    const auto yi = static_cast<std::int32_t>(y);
    const npy_intp pixel_index = y * width + x;
    switch (side) {
        case 0: return {pixel_index, 0, 0, {xi, yi}, {static_cast<std::int32_t>(xi + 1), yi}};
        case 1:
            return {pixel_index,
                    1,
                    1,
                    {static_cast<std::int32_t>(xi + 1), yi},
                    {static_cast<std::int32_t>(xi + 1), static_cast<std::int32_t>(yi + 1)}};
        case 2:
            return {pixel_index,
                    2,
                    2,
                    {static_cast<std::int32_t>(xi + 1), static_cast<std::int32_t>(yi + 1)},
                    {xi, static_cast<std::int32_t>(yi + 1)}};
        default:
            return {pixel_index,
                    3,
                    3,
                    {xi, static_cast<std::int32_t>(yi + 1)},
                    {xi, yi}};
    }
}

template <typename T>
inline bool boundary_side(
    const T* data,
    npy_intp height,
    npy_intp width,
    npy_intp y,
    npy_intp x,
    std::uint8_t side,
    std::uint64_t label
) {
    if (static_cast<std::uint64_t>(data[y * width + x]) != label) {
        return false;
    }
    switch (side) {
        case 0:
            return y == 0 || static_cast<std::uint64_t>(data[(y - 1) * width + x]) != label;
        case 1:
            return x + 1 == width || static_cast<std::uint64_t>(data[y * width + x + 1]) != label;
        case 2:
            return y + 1 == height || static_cast<std::uint64_t>(data[(y + 1) * width + x]) != label;
        default:
            return x == 0 || static_cast<std::uint64_t>(data[y * width + x - 1]) != label;
    }
}

template <typename T>
bool candidate_from_vertex(
    const T* data,
    npy_intp height,
    npy_intp width,
    std::uint64_t label,
    const Vertex& vertex,
    std::uint8_t dir,
    BoundaryEdge& out
) {
    npy_intp y = 0;
    npy_intp x = 0;
    std::uint8_t side = 0;

    switch (dir) {
        case 0:  // east: interior pixel is below the edge
            y = vertex.y;
            x = vertex.x;
            side = 0;
            break;
        case 1:  // south: interior pixel is left of the edge
            y = vertex.y;
            x = static_cast<npy_intp>(vertex.x) - 1;
            side = 1;
            break;
        case 2:  // west: interior pixel is above the edge
            y = static_cast<npy_intp>(vertex.y) - 1;
            x = static_cast<npy_intp>(vertex.x) - 1;
            side = 2;
            break;
        default:  // north: interior pixel is right of the edge
            y = static_cast<npy_intp>(vertex.y) - 1;
            x = vertex.x;
            side = 3;
            break;
    }

    if (y < 0 || y >= height || x < 0 || x >= width) {
        return false;
    }
    if (!boundary_side(data, height, width, y, x, side, label)) {
        return false;
    }
    out = edge_for_pixel(y, x, width, side);
    return true;
}

std::int64_t signed_area2(const std::vector<Vertex>& points) {
    std::int64_t area2 = 0;
    const std::size_t n = points.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Vertex& a = points[i];
        const Vertex& b = points[(i + 1) % n];
        area2 += static_cast<std::int64_t>(a.x) * static_cast<std::int64_t>(b.y) -
                 static_cast<std::int64_t>(b.x) * static_cast<std::int64_t>(a.y);
    }
    return area2;
}

template <typename T>
void trace_loop(
    const T* data,
    npy_intp height,
    npy_intp width,
    std::uint64_t label,
    BoundaryEdge seed,
    std::vector<std::uint8_t>& visited,
    std::vector<Loop>& loops
) {
    Loop loop;
    loop.label = label;
    loop.points.reserve(64);
    const Vertex start = seed.a;
    BoundaryEdge edge = seed;

    for (;;) {
        const auto bit = static_cast<std::uint8_t>(1U << edge.side);
        if ((visited[edge.pixel_index] & bit) != 0) {
            throw std::runtime_error("encountered an already visited contour edge");
        }
        visited[edge.pixel_index] = static_cast<std::uint8_t>(visited[edge.pixel_index] | bit);
        loop.points.push_back(edge.a);

        if (same_vertex(edge.b, start)) {
            break;
        }

        // Interior stays on the right. Right/straight/left/back ordering makes
        // corner-touching pixels separate 4-connected components.
        const std::uint8_t dirs[4] = {
            static_cast<std::uint8_t>((edge.dir + 1U) & 3U),
            edge.dir,
            static_cast<std::uint8_t>((edge.dir + 3U) & 3U),
            static_cast<std::uint8_t>((edge.dir + 2U) & 3U),
        };

        bool found = false;
        for (const std::uint8_t dir : dirs) {
            BoundaryEdge candidate{};
            if (!candidate_from_vertex(data, height, width, label, edge.b, dir, candidate)) {
                continue;
            }
            const auto candidate_bit = static_cast<std::uint8_t>(1U << candidate.side);
            if ((visited[candidate.pixel_index] & candidate_bit) != 0) {
                continue;
            }
            edge = candidate;
            found = true;
            break;
        }
        if (!found) {
            throw std::runtime_error("failed to continue contour");
        }
    }

    loop.is_hole = signed_area2(loop.points) < 0;
    loops.push_back(std::move(loop));
}

template <typename T>
void trace_all(
    const T* data,
    npy_intp height,
    npy_intp width,
    std::vector<Loop>& loops
) {
    const auto n_pixels = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    std::vector<std::uint8_t> visited(n_pixels, 0);

    for (npy_intp y = 0; y < height; ++y) {
        for (npy_intp x = 0; x < width; ++x) {
            const npy_intp pixel_index = y * width + x;
            const std::uint64_t label = static_cast<std::uint64_t>(data[pixel_index]);
            if (label == 0) {
                continue;
            }
            for (std::uint8_t side = 0; side < 4; ++side) {
                const auto bit = static_cast<std::uint8_t>(1U << side);
                if ((visited[pixel_index] & bit) != 0) {
                    continue;
                }
                if (!boundary_side(data, height, width, y, x, side, label)) {
                    continue;
                }
                trace_loop(
                    data,
                    height,
                    width,
                    label,
                    edge_for_pixel(y, x, width, side),
                    visited,
                    loops
                );
            }
        }
    }

    std::stable_sort(loops.begin(), loops.end(), [](const Loop& a, const Loop& b) {
        return a.label < b.label;
    });
}

PyObject* py_contours2d(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return nullptr;
    }
    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "labels must be a NumPy array");
        return nullptr;
    }

    auto* arr = reinterpret_cast<PyArrayObject*>(obj);
    if (PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "labels must be a 2D array");
        return nullptr;
    }
    if (!PyArray_ISCARRAY_RO(arr) || !PyArray_ISNBO(PyArray_DESCR(arr)->byteorder)) {
        PyErr_SetString(PyExc_ValueError, "labels must be C-contiguous, aligned, and native-endian");
        return nullptr;
    }

    const int typenum = PyArray_TYPE(arr);
    if (typenum != NPY_UINT32 && typenum != NPY_UINT64) {
        PyErr_SetString(PyExc_TypeError, "labels dtype must be uint32 or uint64");
        return nullptr;
    }

    const npy_intp height = PyArray_DIM(arr, 0);
    const npy_intp width = PyArray_DIM(arr, 1);
    if (height > std::numeric_limits<std::int32_t>::max() - 1LL ||
        width > std::numeric_limits<std::int32_t>::max() - 1LL) {
        PyErr_SetString(PyExc_OverflowError, "array dimensions exceed int32 coordinate range");
        return nullptr;
    }
    if (height != 0 && static_cast<std::size_t>(width) >
                           std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(height)) {
        PyErr_NoMemory();
        return nullptr;
    }

    std::vector<Loop> loops;
    enum class CoreError { none, bad_alloc, unexpected };
    CoreError core_error = CoreError::none;
    std::string error_message;

    Py_BEGIN_ALLOW_THREADS
    try {
        if (typenum == NPY_UINT32) {
            trace_all(
                reinterpret_cast<const std::uint32_t*>(PyArray_DATA(arr)), height, width, loops
            );
        } else {
            trace_all(
                reinterpret_cast<const std::uint64_t*>(PyArray_DATA(arr)), height, width, loops
            );
        }
    } catch (const std::bad_alloc&) {
        core_error = CoreError::bad_alloc;
    } catch (const std::exception& exc) {
        core_error = CoreError::unexpected;
        try {
            error_message = exc.what();
        } catch (...) {
            error_message.clear();
        }
    } catch (...) {
        core_error = CoreError::unexpected;
    }
    Py_END_ALLOW_THREADS

    if (core_error == CoreError::bad_alloc) {
        PyErr_NoMemory();
        return nullptr;
    }
    if (core_error == CoreError::unexpected) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error_message.empty() ? "unexpected C++ error while tracing contours" : error_message.c_str()
        );
        return nullptr;
    }

    std::vector<std::uint64_t> ids;
    std::vector<npy_intp> label_offsets;
    label_offsets.push_back(0);
    std::size_t total_points = 0;
    std::uint64_t last_label = 0;
    bool first = true;
    for (std::size_t i = 0; i < loops.size(); ++i) {
        if (first || loops[i].label != last_label) {
            if (!first) {
                label_offsets.push_back(static_cast<npy_intp>(i));
            }
            ids.push_back(loops[i].label);
            last_label = loops[i].label;
            first = false;
        }
        if (loops[i].points.size() > std::numeric_limits<std::size_t>::max() - total_points) {
            PyErr_NoMemory();
            return nullptr;
        }
        total_points += loops[i].points.size();
    }
    if (!first) {
        label_offsets.push_back(static_cast<npy_intp>(loops.size()));
    }
    if (total_points > static_cast<std::size_t>(std::numeric_limits<npy_intp>::max())) {
        PyErr_NoMemory();
        return nullptr;
    }

    npy_intp ids_dim[1] = {static_cast<npy_intp>(ids.size())};
    npy_intp points_dims[2] = {static_cast<npy_intp>(total_points), 2};
    npy_intp contour_offsets_dim[1] = {static_cast<npy_intp>(loops.size() + 1)};
    npy_intp label_offsets_dim[1] = {static_cast<npy_intp>(label_offsets.size())};
    npy_intp holes_dim[1] = {static_cast<npy_intp>(loops.size())};

    PyObject* ids_arr = PyArray_SimpleNew(1, ids_dim, typenum);
    PyObject* points_arr = PyArray_SimpleNew(2, points_dims, NPY_INT32);
    PyObject* contour_offsets_arr = PyArray_SimpleNew(1, contour_offsets_dim, NPY_INTP);
    PyObject* label_offsets_arr = PyArray_SimpleNew(1, label_offsets_dim, NPY_INTP);
    PyObject* holes_arr = PyArray_SimpleNew(1, holes_dim, NPY_BOOL);
    if (!ids_arr || !points_arr || !contour_offsets_arr || !label_offsets_arr || !holes_arr) {
        Py_XDECREF(ids_arr);
        Py_XDECREF(points_arr);
        Py_XDECREF(contour_offsets_arr);
        Py_XDECREF(label_offsets_arr);
        Py_XDECREF(holes_arr);
        return nullptr;
    }

    if (typenum == NPY_UINT32) {
        auto* out = reinterpret_cast<std::uint32_t*>(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(ids_arr))
        );
        for (std::size_t i = 0; i < ids.size(); ++i) {
            out[i] = static_cast<std::uint32_t>(ids[i]);
        }
    } else {
        auto* out = reinterpret_cast<std::uint64_t*>(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(ids_arr))
        );
        std::copy(ids.begin(), ids.end(), out);
    }

    auto* out_points = reinterpret_cast<std::int32_t*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(points_arr))
    );
    auto* out_contour_offsets = reinterpret_cast<npy_intp*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(contour_offsets_arr))
    );
    auto* out_label_offsets = reinterpret_cast<npy_intp*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(label_offsets_arr))
    );
    auto* out_holes = reinterpret_cast<npy_bool*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(holes_arr))
    );

    npy_intp point_offset = 0;
    out_contour_offsets[0] = 0;
    for (std::size_t i = 0; i < loops.size(); ++i) {
        for (const Vertex& v : loops[i].points) {
            out_points[2 * point_offset] = v.x;
            out_points[2 * point_offset + 1] = v.y;
            ++point_offset;
        }
        out_contour_offsets[i + 1] = point_offset;
        out_holes[i] = loops[i].is_hole ? 1 : 0;
    }
    std::copy(label_offsets.begin(), label_offsets.end(), out_label_offsets);

    PyObject* result = PyTuple_New(5);
    if (!result) {
        Py_DECREF(ids_arr);
        Py_DECREF(points_arr);
        Py_DECREF(contour_offsets_arr);
        Py_DECREF(label_offsets_arr);
        Py_DECREF(holes_arr);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, ids_arr);
    PyTuple_SET_ITEM(result, 1, points_arr);
    PyTuple_SET_ITEM(result, 2, contour_offsets_arr);
    PyTuple_SET_ITEM(result, 3, label_offsets_arr);
    PyTuple_SET_ITEM(result, 4, holes_arr);
    return result;
}

PyMethodDef methods[] = {
    {"contours2d", py_contours2d, METH_VARARGS, "Extract pixel-edge contours from a label image."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "fastlabelcontours C++ core",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__core(void) {
    import_array();
    return PyModule_Create(&module);
}
