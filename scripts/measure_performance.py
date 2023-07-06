import argparse
import json
import os
import time

import torch
import torch.nn as nn

try:
    import ggml_conversion as ggml
except (ImportError, ModuleNotFoundError):
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import ggml_conversion as ggml


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-iters", type=int, default=1_000)
    parser.add_argument("--save-dir", type=str, default="models")
    return parser.parse_args()


class ResNetBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(ResNetBlock, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return x + self.f(x)


class ResNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int):
        super(ResNet, self).__init__()
        self.f = nn.Sequential(
            *[ResNetBlock(input_dim, hidden_dim) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.f(x)


def time_model(model: nn.Module | ggml.GGMLModel, X: torch.Tensor, num_iters: int) -> float:
    if isinstance(model, nn.Module):
        model.eval()
    start = time.time()
    for _ in range(num_iters):
        model(X)
    end = time.time()
    return (end - start) / num_iters


def _to_float_vector(model: ggml.GGMLModel, X: torch.Tensor):
    return getattr(model._module, "FloatVector")(X.flatten().tolist())


def time_ggml_model(model: ggml.GGMLModel, X: torch.Tensor, num_iters: int) -> float:
    Xv = _to_float_vector(model, X)
    start = time.time()
    for _ in range(num_iters):
        model.model.forward(Xv)
    end = time.time()
    return (end - start) / num_iters


def write_jsonl(filename: str, data: list[dict]):
    append = os.path.exists(filename)
    with open(filename, "a" if append else "w") as f:
        for d in data:
            json.dump(d, f)
            f.write("\n")


def main():
    args = get_args()
    print("Input dim:", args.input_dim)
    print("Hidden dim:", args.hidden_dim)
    print("Num blocks:", args.num_blocks)
    print("Batch size:", args.batch_size)
    print("Num iters:", args.num_iters)
    print("Save dir:", args.save_dir)
    print()

    print("Creating model...")
    torch_model = ResNet(args.input_dim, args.hidden_dim, args.num_blocks)
    X = torch.randn(args.batch_size, args.input_dim)

    print("Converting model...")
    ggml_model = ggml.convert(torch_model, (X,), args.save_dir)

    print("Timing torch model...")
    torch_time = time_model(torch_model, X, args.num_iters)

    print("Timing ggml model...")
    ggml_time = time_model(ggml_model, X, args.num_iters)

    print("Timing ggml model (via .forward)...")
    ggml_time2 = time_ggml_model(ggml_model, X, args.num_iters)

    print(f"torch: {torch_time:.4f} s")
    print(f"ggml: {ggml_time:.4f} s")
    print(f"ggml (via .forward): {ggml_time2:.4f} s")

    filename = os.path.join(args.save_dir, "results.jsonl")
    write_jsonl(
        filename,
        [
            {
                "torch": torch_time,
                "ggml": ggml_time,
                "ggml_forward": ggml_time2,
                "input_dim": args.input_dim,
                "hidden_dim": args.hidden_dim,
                "num_blocks": args.num_blocks,
                "batch_size": args.batch_size,
                "num_iters": args.num_iters,
            }
        ]
    )


if __name__ == "__main__":
    main()
