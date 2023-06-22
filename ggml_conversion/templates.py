from typing import Final

MAIN: Final[str] = """#include <stdio.h>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ggml/ggml.h"

#include "common.h"
#include "common-ggml.h"

namespace py = pybind11;

struct ggml_tensor * ggml_linear(
    struct ggml_context *ctx,
    struct ggml_tensor *input,
    struct ggml_tensor *weight,
    struct ggml_tensor *bias
) {{
    // Wx + b
    struct ggml_tensor *wx = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, input)), weight);
    return ggml_add(ctx, wx, ggml_repeat(ctx, wx, ggml_repeat(ctx, ggml_reshape_2d(ctx, bias, 1, bias->ne[0]), wx)));
}}

struct ggml_model {{
    {model_struct}
}};

void ggml_set_f32_2d(struct ggml_tensor *tensor, int i, int j, float value) {{
    *(float *) ((char *) tensor->data + i*tensor->nb[0] + j*tensor->nb[1]) = value;
}}

void ggml_set_f32_3d(struct ggml_tensor *tensor, int i, int j, int k, float value) {{
    *(float *) ((char *) tensor->data + i*tensor->nb[0] + j*tensor->nb[1] + k*tensor->nb[2]) = value;
}}

void ggml_set_f32_4d(struct ggml_tensor *tensor, int i, int j, int k, int l, float value) {{
    *(float *) ((char *) tensor->data + i*tensor->nb[0] + j*tensor->nb[1] + k*tensor->nb[2] + l*tensor->nb[3]) = value;
}}

struct ggml_tensor * model_forward(struct ggml_context *ctx, struct ggml_tensor *input, struct ggml_model *model) {{
    {forward}
}}

struct ggml_tensor * set_input(struct ggml_context *ctx, const {input_type}& input) {{
    {set_input}
}}

{output_type} get_output(struct ggml_tensor *output) {{
    {get_output}
}}

class Model {{
public:
    Model() {{
        struct ggml_init_params p = {{
            .mem_size = {mem_size},
            .mem_buffer = NULL,
        }};
        ctx = ggml_init(p);
        model = {{
            {model_init}
        }};
    }}

    void set_weights({input_args}) {{
        {set_weights}
    }}

    {output_type} forward(const {input_type}& input) {{
        struct ggml_tensor *input_tensor = set_input(ctx, input);
        struct ggml_tensor *output = model_forward(ctx, input_tensor, &model);
        struct ggml_cgraph graph = ggml_build_forward(output);
        ggml_graph_compute(ctx, &graph);
        return get_output(output);
    }}

private:
    struct ggml_context *ctx;
    struct ggml_model model;
}};

PYBIND11_MODULE({model_name}, m) {{
    py::class_<Model>(m, "Model")
        .def(py::init<>())
        .def("set_weights", &Model::set_weights)
        .def("forward", &Model::forward);
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
