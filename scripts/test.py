import subprocess
import sys
from pathlib import Path
import pytest

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

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 10)


class AddMatrix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.rand([5, 5])

    def forward(self, x):
        return x + self.w

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(5, 5)


class MatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.rand([5, 5])

    def forward(self, x):
        return torch.matmul(x, self.w)

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 5)


class LinearProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.rand([5, 5])
        self.b = torch.rand([5])

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 5)


class LinearLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(5, 5)

    def forward(self, x):
        return self.layer(x)

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 5)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(5, 5)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        return self.activation(self.layer(x))

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 5)


@pytest.mark.parametrize(
    "model", [AddConst, AddMatrix, MatMul, LinearProjection, LinearLayer, MLP]
)
def test_model(Model):
    directory = Path(sys.argv[1]) / Model.__name__
    directory.mkdir(exist_ok=True, parents=True)
    torch.onnx.export(
        Model(),
        Model.get_dummy_input_tensor(),
        str(directory / "model.onnx"),
    )
    model = onnx.load(str(directory / "model.onnx"))
    conversion = core.Conversion.from_onnx_model(model)
    conversion.build(directory)
    subprocess.run([directory / "build" / conversion.name], check=True)
