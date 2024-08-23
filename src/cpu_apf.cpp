#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <queue>
#include <map>

namespace py = pybind11;

namespace {

    inline long get_2d(long i, long j, long size_dim1) {
        return j + i * size_dim1;
    }

    struct Point2d {
        long x;
        long y;
    };

    inline float distance_2d(Point2d p1, Point2d p2) {
        return (float) sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
    }

    struct Candidate {
        Point2d p{};
        Point2d ori{};
        float distance{};

        Candidate(long x, long y, float distance) {
            this->p = {x, y};
            this->ori = {x, y};
            this->distance = distance;
        }

        Candidate(Point2d p, Point2d ori, float distance) {
            this->p = p;
            this->ori = ori;
            this->distance = distance;
        }
    };

    const int directions[4][2] = {
            {0,  1},
            {0,  -1},
            {1,  0},
            {-1, 0},
    };

    std::tuple<py::array_t<float>, bool> cpu_apf(py::array_t<std::uint8_t> &input) {
        // Parse the inputs
        py::buffer_info buf_in = input.request();
        auto result = py::array_t<float>(buf_in.shape);
        py::buffer_info buf_out = result.request();
        auto *ptr_in = (std::uint8_t *) buf_in.ptr;
        auto *ptr_out = (float *) buf_out.ptr;

        const auto size_dim0 = input.shape(0);
        const auto size_dim1 = input.shape(1);


        std::queue<Candidate> candidates;
        bool visited[size_dim0 * size_dim1];
        bool is_empty = true;

        for (std::int64_t i = 0; i < size_dim0; ++i) {
            for (std::int64_t j = 0; j < size_dim1; ++j) {
                long aim = get_2d(i, j, size_dim1);
                ptr_out[aim] = 0.f;
                if (ptr_in[aim]) {
                    is_empty = false;
                    candidates.emplace(i, j, 0.);
                    visited[aim] = true;
                } else {
                    visited[aim] = false;
                }
            }
        }
        while (not candidates.empty()) {
            auto candidate = candidates.front();
            candidates.pop();
            for (auto &direction: directions) {
                Point2d new_position = {candidate.p.x + direction[0], candidate.p.y + direction[1]};
                long aim = get_2d(new_position.x, new_position.y, size_dim1);
                if (0 <= aim and aim < size_dim0 * size_dim1 and not visited[aim]) {
                    float new_distance = distance_2d(candidate.ori, new_position);
                    candidates.emplace(new_position, candidate.ori, new_distance);
                    visited[aim] = true;
                    ptr_out[aim] = new_distance;
                }
            }
        }
        return std::make_tuple(std::move(result), is_empty);
    }

    PYBIND11_MODULE(cpu_apf, m
    ) {
        m.def("cpu_apf_bool", &cpu_apf);
    }

}  // namespace