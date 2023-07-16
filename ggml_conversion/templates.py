from typing import Final

MAIN: Final[str] = """#include <stdio.h>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "ggml/ggml.h"

#include "common.h"
#include "common-ggml.h"

PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, std::vector<float>>);

namespace py = pybind11;

struct ggml_tensor * ggml_linear(
    struct ggml_context *ctx,
    struct ggml_tensor *input,
    struct ggml_tensor *weight,
    struct ggml_tensor *bias
) {{
    // Wx + b
    struct ggml_tensor *wx = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, input, weight)));
    return ggml_add(ctx, wx, ggml_repeat(ctx, ggml_reshape_2d(ctx, bias, bias->ne[0], 1), wx));
}}

void ggml_vec_eq_f32(const int n, float *out, const float *a, const float *b) {{
    for (int i = 0; i < n; i++) {{
        out[i] = a[i] == b[i] ? 1.0f : 0.0f;
    }}
}}

struct ggml_tensor * ggml_equal(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {{
    return ggml_map_binary_f32(ctx, a, b, &ggml_vec_eq_f32);
}}

struct ggml_tensor * ggml_ones_like(struct ggml_context *ctx, struct ggml_tensor *a) {{
    struct ggml_tensor *ones = ggml_dup_tensor(ctx, a);
    ggml_set_f32(ones, 1.0f);
    return ones;
}}

struct ggml_tensor * ggml_where(struct ggml_context *ctx, struct ggml_tensor *cond, struct ggml_tensor *a, struct ggml_tensor *b) {{
    // p * a + (1 - p) * b
    struct ggml_tensor *ones = ggml_ones_like(ctx, cond);
    struct ggml_tensor *pa = ggml_mul(ctx, cond, a);
    struct ggml_tensor *np = ggml_sub(ctx, ones, cond);
    struct ggml_tensor *pb = ggml_mul(ctx, np, b);
    return ggml_add(ctx, pa, pb);
}}

struct ggml_model {{
    {model_struct}
}};

struct ggml_tensor * model_forward(struct ggml_context *ctx, struct ggml_tensor *input, struct ggml_model *model) {{
    {forward}
}}

void set_tensor(struct ggml_tensor *tensor, const std::vector<float>& data) {{
    std::memcpy(tensor->data, data.data(), ggml_nbytes(tensor));
}}

struct ggml_tensor * set_input(struct ggml_context *ctx, const std::vector<float>& input) {{
    struct ggml_tensor *input_tensor = {input_tensor};
    set_tensor(input_tensor, input);
    return input_tensor;
}}

int64_t ggml_tensor_size(struct ggml_tensor *tensor) {{
    return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}}

std::vector<float> get_output(struct ggml_tensor *output) {{
    std::vector<float> result(ggml_tensor_size(output));
    std::memcpy(result.data(), output->data, ggml_nbytes(output));
    return result;
}}

class {camel_case_model_name} {{
public:
    {camel_case_model_name}() {{
        struct ggml_init_params p = {{
            .mem_size = {model_mem_size},
            .mem_buffer = NULL,
            .no_alloc = false,
        }};
        ctx = ggml_init(p);
        model = {{
            {model_init}
        }};
    }}

    ~{camel_case_model_name}() {{
        ggml_free(ctx);
    }}

    py::tuple output_shape() {{
        const auto shape = std::make_tuple({output_shape});
        return py::cast(shape);
    }}

    void set_weights(std::map<std::string, std::vector<float>>& weights) {{
        {set_weights}
    }}

    std::vector<float> forward(const std::vector<float>& input) {{
        struct ggml_init_params p = {{
            .mem_size = {eval_mem_size},
            .mem_buffer = NULL,
        }};
        struct ggml_context *ctx0 = ggml_init(p);
        struct ggml_tensor *input_tensor = set_input(ctx0, input);
        struct ggml_tensor *output = model_forward(ctx0, input_tensor, &model);
        struct ggml_cgraph graph = ggml_build_forward(output);
        ggml_graph_compute(ctx0, &graph);
        const auto result = get_output(output);
        ggml_free(ctx0);
        return result;
    }}

private:
    struct ggml_context *ctx;
    struct ggml_model model;
}};

PYBIND11_MODULE({model_name}, m) {{
    py::class_<{camel_case_model_name}>(m, "{camel_case_model_name}")
        .def(py::init<>())
        .def("output_shape", &{camel_case_model_name}::output_shape)
        .def("set_weights", &{camel_case_model_name}::set_weights)
        .def("forward", &{camel_case_model_name}::forward);
    py::bind_vector<std::vector<float>>(m, "FloatVector");
    py::bind_map<std::map<std::string, std::vector<float>>>(m, "FloatMap");
}}
"""

CMAKELISTS: Final[str] = """cmake_minimum_required(VERSION 3.24)

set(PROJECT_NAME {model_name})
project(${{PROJECT_NAME}})

include_directories(ggml/include)

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(ggml)
add_subdirectory(pybind11)

pybind11_add_module(${{PROJECT_NAME}} main.cpp)
target_include_directories(${{PROJECT_NAME}} PUBLIC ggml/include ggml/examples)
target_link_libraries(${{PROJECT_NAME}} PRIVATE ggml pybind11::module)
"""
