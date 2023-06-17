from typing import Final

MAIN: Final[str] = """#include <stdio.h>
#include <vector>

#include "ggml/ggml.h"

#include "common.h"
#include "common-ggml.h"

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

struct ggml_tensor * forward(struct ggml_context *ctx, struct ggml_tensor *input, struct ggml_model *model) {{
    {forward}
}}

int main() {{
    struct ggml_init_params p = {{
        .mem_size = {mem_size},
        .mem_buffer = NULL,
    }};
    struct ggml_context *ctx = ggml_init(p);

    struct ggml_tensor *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, {dim_0}, {dim_1});
    struct ggml_model model = {{
        {model_init}
    }};
    struct ggml_tensor *output = forward(ctx, input, &model);
    struct ggml_cgraph graph = ggml_build_forward(output);

    {set_input}

    ggml_graph_compute(ctx, &graph);
    printf("output = %f\\n", ggml_get_f32_1d(output, 0));
}}
"""

CMAKELISTS: Final[str] = """cmake_minimum_required(VERSION 3.24)

set(PROJECT_NAME {model_name})
project(${{PROJECT_NAME}})

include_directories(ggml/include)

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(ggml)

add_executable(${{PROJECT_NAME}} main.cpp)
target_include_directories(${{PROJECT_NAME}} PUBLIC ggml/include ggml/examples)
target_link_libraries(${{PROJECT_NAME}} PRIVATE ggml)
"""
