#include "ggml/ggml.h"

#include <stdio.h>

int main() {
    struct ggml_init_params p = {
        .mem_size = 1024 * 1024 * 1024,
        .mem_buffer = NULL,
    };
    struct ggml_context *ctx = ggml_init(p);

    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 5);
    struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 5);
    struct ggml_tensor *output = ggml_repeat(ctx, a, b);
    struct ggml_cgraph graph = ggml_build_forward(output);

    ggml_set_f32(a, 1.0);
    ggml_set_f32(b, 1.0);

    ggml_graph_compute(ctx, &graph);
    printf("shape = (%d, %d)\n", output->ne[0], output->ne[1]);
}
