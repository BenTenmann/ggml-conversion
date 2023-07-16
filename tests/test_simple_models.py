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


class MLP3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(5, 5)
        self.layer2 = torch.nn.Linear(5, 5)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        return self.activation(self.layer2(self.activation(self.layer1(x))))

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 5)


class Transpose(torch.nn.Module):
    def forward(self, x):
        return x.T

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 5)


class Erf(torch.nn.Module):
    def forward(self, x):
        return torch.erf(x)

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 5)


class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.reshape(-1)

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 5)


class ConstantOfShape(torch.nn.Module):
    # TODO: we need to find a way to test this
    # currently could not find a torch operation which would translate into ConstantOfShape
    # without additional operators
    def forward(self, x):
        return x + 1

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(2,)


class Pow(torch.nn.Module):
    def forward(self, x):
        return torch.pow(x, 3)

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randn(3, 5)


class WhereEqual(torch.nn.Module):
    def forward(self, x):
        return torch.where(x == 0.0, torch.ones_like(x), torch.zeros_like(x))

    @classmethod
    def get_dummy_input_tensor(cls):
        return torch.randint(2, size=(3, 5), dtype=torch.float32)


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
    MLP3,
    Transpose,
    Erf,
    Reshape,
    # ConstantOfShape,
    Pow,
    WhereEqual,
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
    ggml_model = core.convert(torch_model, (X,), directory)

    torch_result = torch_model(X)
    ggml_result = ggml_model(X)
    assert torch.allclose(torch_result, ggml_result, atol=1e-6)
