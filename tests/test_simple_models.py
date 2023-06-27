from pathlib import Path
import pytest
import time

import torch
import torch.onnx

from ggml_conversion import core

TEST_DATA_DIR = Path(__file__).parents[1] / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)
(TEST_DATA_DIR / ".gitignore").write_text("*\n")


class AddConst(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = 5

    def forward(self, x):
        return x + self.w

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 10)


class MulConst(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = 5

    def forward(self, x):
        return x * self.w

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 10)


class AddMatrix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randn([5, 5]) * 5

    def forward(self, x):
        return x + self.w

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(5, 5)


class MulMatrix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.ones([5, 5]) * 5

    def forward(self, x):
        return x * self.w

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(5, 5)


class Arithmetic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.ones([5, 5]) * 5
        self.b = torch.ones([5])

    def forward(self, x):
        return x * self.w + self.w - self.b

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(5, 5)


class MatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.ones([5, 5])

    def forward(self, x):
        return torch.matmul(x, self.w)

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 5)


class LinearProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.ones([5, 5])
        self.b = torch.ones([5])

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
        self.layer = torch.nn.Linear(5, 2)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        return self.activation(self.layer(x))

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 5)


class MLP2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(5, 5)
        self.layer2 = torch.nn.Linear(5, 5)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        return self.activation(self.layer2(self.activation(self.layer1(x))))

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 5)


@pytest.fixture(scope="session", autouse=True)
def timestamp():
    yield time.strftime("%Y%m%d-%H%M%S")


MODELS = [
    AddConst,
    MulConst,
    AddMatrix,
    MulMatrix,
    Arithmetic,
    MatMul,
    LinearProjection,
    LinearLayer,
    MLP,
    MLP2,
]


@pytest.mark.parametrize("Model", MODELS)
def test_model_runs(Model):
    result = Model()(Model.get_dummy_input_tensor())
    assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize("Model", MODELS)
def test_model_builds(Model, timestamp):
    directory = TEST_DATA_DIR / timestamp / Model.__name__
    directory.mkdir(exist_ok=True, parents=True)
    X = Model.get_dummy_input_tensor()
    torch_model = Model()
    ggml_model = core.run_ggml_converter(torch_model, (X,), directory)

    torch_result = torch_model(X)
    ggml_result = ggml_model(X)
    assert torch.allclose(torch_result, ggml_result, atol=1e-6)
