import subprocess
import sys
from pathlib import Path

import onnx
import torch
import torch.onnx

from ggml_conversion import core


class AddConst(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = 5

    def forward(self, x):
        return x + self.w


class AddMatrix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.rand([5, 5])

    def forward(self, x):
        return x + self.w


class MatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.rand([5, 5])

    def forward(self, x):
        return torch.matmul(x, self.w)


class LinearProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.rand([5, 5])
        self.b = torch.rand([5])

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b


class LinearLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(5, 5)

    def forward(self, x):
        return self.layer(x)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(5, 5)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        return self.activation(self.layer(x))


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(exist_ok=True, parents=True)
    torch.onnx.export(
        Net(),
        torch.randn(3, 10),
        str(directory / "model.onnx"),
    )
    model = onnx.load(str(directory / "model.onnx"))
    conversion = core.Conversion.from_onnx_model(model)
    conversion.build(directory)
    subprocess.run([directory / "build" / conversion.name], check=True)


if __name__ == "__main__":
    main()
