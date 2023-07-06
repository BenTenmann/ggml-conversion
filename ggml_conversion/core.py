import collections
import importlib
import logging
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final, Literal

import git
import inflection
import onnx
import pydantic
import torch

from ggml_conversion import modules, templates, utils

LOGGER: Final[logging.Logger] = utils.get_logger(__name__)

Dependency = Literal["ggml", "pybind11"]

GGML_REPO: Final[str] = "https://github.com/BenTenmann/ggml.git"
PYBIND11_REPO: Final[str] = "https://github.com/pybind/pybind11.git"
REPOS: Final[dict[Dependency, str]] = {
    "ggml": GGML_REPO,
    "pybind11": PYBIND11_REPO,
}
CACHE_DIR: Final[Path] = Path.home() / ".cache" / "ggml-conversion"

FLOAT_BYTES: Final[int] = 4

# not sure if this is correct (taken from https://github.com/ggerganov/ggml/blob/master/examples/gpt-2/main.cpp#L194)
OBJECT_OVERHEAD: Final[int] = 512


def create_io_name_map(model: onnx.ModelProto) -> dict[str, Literal["input", "output"]]:
    # TODO: handle multiple inputs and outputs
    # input and output names are often not allowed in C
    mapping: dict[str, Literal["input", "output"]] = {}
    if len(model.graph.input) > 0:
        mapping[model.graph.input[0].name] = "input"
    if len(model.graph.output) > 0:
        mapping[model.graph.output[0].name] = "output"
    return mapping


def get_tensors(model: onnx.ModelProto) -> dict[str, modules.Tensor]:
    out = {}
    for init in model.graph.initializer:
        name = modules.reformat_name(init.name, create_io_name_map(model))
        shape = tuple(init.dims)
        out[name] = modules.Tensor(name=name, shape=shape)
    for inp in model.graph.input:
        name = modules.reformat_name(inp.name, create_io_name_map(model))
        shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        out[name] = modules.Tensor(name=name, shape=shape)
    for value in onnx.shape_inference.infer_shapes(model).graph.value_info:
        name = modules.reformat_name(value.name, create_io_name_map(model))
        shape = tuple(d.dim_value for d in value.type.tensor_type.shape.dim)
        out[name] = modules.Tensor(name=name, shape=shape)
    for output in model.graph.output:
        name = modules.reformat_name(output.name, create_io_name_map(model))
        shape = tuple(d.dim_value for d in output.type.tensor_type.shape.dim)
        out[name] = modules.Tensor(name=name, shape=shape)
    return out


def generate_forward(model: onnx.ModelProto) -> str:
    io_name_map = create_io_name_map(model)
    mods: list[str] = []
    for node in model.graph.node:
        if node.op_type not in modules.GGML_OPERATORS:
            raise NotImplementedError(f"Operator {node.op_type} is not supported")
        mods.append(
            modules.GGML_OPERATORS[node.op_type](
                node=node,
                graph=model.graph,
                tensors=get_tensors(model),
                io_names_map=io_name_map,
            ).convert()
        )
    mods.append(
        "return {};".format(
            modules.reformat_name(
                model.graph.output[0].name,
                io_names_map=io_name_map
            )
        )
    )
    return "\n    ".join(mods)


def generate_model_struct(model: onnx.ModelProto) -> str:
    return "\n    ".join(
        "struct ggml_tensor *{};".format(
            modules.reformat_name(init.name, create_io_name_map(model))
        )
        for init in model.graph.initializer
    )


def generate_model_init(model: onnx.ModelProto) -> str:
    weights = [
        init.name for init in model.graph.initializer
    ]
    dimension = [
        init.dims for init in model.graph.initializer
    ]
    return "\n        ".join(
        ".{name} = ggml_new_tensor_{n}d(ctx, GGML_TYPE_F32, {dims}),".format(
            name=modules.reformat_name(name, io_names_map=create_io_name_map(model)),
            n=len(dims),
            dims=", ".join(str(d) for d in list(dims)[::-1])
        )
        for name, dims in zip(weights, dimension)
    )


def generate_model_mem_size(model: onnx.ModelProto) -> str:
    return str(
        sum(
            FLOAT_BYTES * math.prod(init.dims) + OBJECT_OVERHEAD
            for init in model.graph.initializer
        )
    )


def generate_input_tensor(model: onnx.ModelProto) -> str:
    if len(model.graph.input) != 1:
        raise NotImplementedError("Only models with one input are supported")
    value = model.graph.input[0]
    return "ggml_new_tensor_{ndim}d(ctx, GGML_TYPE_F32, {shape})".format(
        ndim=len(list(value.type.tensor_type.shape.dim)),
        shape=", ".join(str(d.dim_value) for d in list(value.type.tensor_type.shape.dim)[::-1]),
    )


def generate_set_weights(model: onnx.ModelProto) -> str:
    return "\n    ".join(
        "set_tensor(model.{name}, weights[\"{unformatted_name}\"]);".format(
            name=modules.reformat_name(init.name, create_io_name_map(model)),
            unformatted_name=init.name,
        )
        for init in model.graph.initializer
    )


class GGMLModel:
    def __init__(self, model_path: str, model_name: str):
        # since we are importing the module, if we try to build another model of the same name
        # in the same runtime, it will not update the model
        # this is because Python caches imports
        if model_path in sys.modules:
            module = importlib.reload(sys.modules[model_path])
        else:
            module = importlib.import_module(model_path)
        self._module = module
        self.model = getattr(module, model_name)()
        self.model_name = model_name

    def set_weights(self, weights: collections.OrderedDict[str, torch.Tensor]) -> None:
        FloatMap = getattr(self._module, "FloatMap")
        FloatVector = getattr(self._module, "FloatVector")
        W = FloatMap()
        for k, v in weights.items():
            W[k] = FloatVector(v.ravel().tolist())
        self.model.set_weights(W)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        FloatVector = getattr(self._module, "FloatVector")
        return torch.tensor(
            list(self.model.forward(FloatVector(x.ravel().tolist()))), dtype=x.dtype, device=x.device
        ).view(*self.model.output_shape())


def get_dependency(dependency: Dependency, build_dir: Path) -> None:
    LOGGER.info(f"Getting dependency {dependency}")
    cache = CACHE_DIR / dependency
    if cache.exists() and cache.is_dir():
        LOGGER.info(f"{dependency} found in cache. Pulling, then copying...")
        git.Repo(cache).remotes.origin.pull()
        shutil.copytree(
            str(cache),
            str(build_dir / dependency),
        )
        return
    LOGGER.info(f"{dependency} not in cache. Cloning into cache...")
    git.Repo.clone_from(REPOS[dependency], cache)
    get_dependency(dependency, build_dir)


class Conversion(pydantic.BaseModel):
    name: str
    main: str = pydantic.Field(
        ...,
        description="The main function of the generated C++ code",
        repr=False,
    )
    cmakelists: str = pydantic.Field(
        ...,
        description="The CMakeLists.txt file of the generated C++ code",
        repr=False,
    )

    def build(self, path: Path | str) -> None:
        LOGGER.info(f"Building in {path}")
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        get_dependency("ggml", path)
        get_dependency("pybind11", path)
        (path / "main.cpp").write_text(self.main)
        (path / "CMakeLists.txt").write_text(self.cmakelists)
        build_dir = path / "build"
        build_dir.mkdir(exist_ok=True, parents=True)
        subprocess.run(["cmake", ".."], cwd=build_dir)
        subprocess.run(["make"], cwd=build_dir)

    @classmethod
    def from_onnx_model(cls, model: onnx.ModelProto, model_name: str) -> "Conversion":
        name = inflection.underscore(model_name)
        return cls(
            name=name,
            main=templates.MAIN.format(
                model_struct=generate_model_struct(model),
                forward=generate_forward(model),
                model_init=generate_model_init(model),
                input_tensor=generate_input_tensor(model),
                model_mem_size=generate_model_mem_size(model),
                eval_mem_size="1024 * 1024 * 1024",
                set_weights=generate_set_weights(model),
                model_name=name,
                output_shape=", ".join(str(d.dim_value) for d in model.graph.output[0].type.tensor_type.shape.dim),
                camel_case_model_name=model_name
            ),
            cmakelists=templates.CMAKELISTS.format(
                model_name=name,
            ),
        )

    @property
    def binary_name(self) -> str:
        version = "".join(platform.python_version_tuple()[:2])
        return f"{self.name}.{platform.python_implementation()}-{version}-{platform.system()}.so".lower()


def convert(
    torch_model: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    save_dir: Path | str | None = None,
) -> GGMLModel:
    """Converts a PyTorch model to a GGML model and builds it.

    Args:
        torch_model (torch.nn.Module): The PyTorch model to convert.
        args (tuple[torch.Tensor, ...]): The arguments to pass to the model.
        save_dir (Path | str | None, optional): The directory to save the built model to. Defaults to None.

    Returns:
        GGMLModel: The built GGML model.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> import torch.nn.functional as F
        >>> import ggml_conversion as ggml
        >>>
        >>> class Net(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc1 = nn.Linear(256, 120)
        ...         self.fc2 = nn.Linear(120, 84)
        ...         self.fc3 = nn.Linear(84, 10)
        ...
        ...     def forward(self, x):
        ...         x = F.relu(self.fc1(x))
        ...         x = F.relu(self.fc2(x))
        ...         x = self.fc3(x)
        ...         return x
        >>>
        >>> net = Net()
        >>> X = torch.randn(100, 256)
        >>> ggml_model = ggml.convert(net, (X,))
        >>> torch.testing.assert_allclose(net(X), ggml_model(X))
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        torch.onnx.export(torch_model, args, str(tmpdir / "model.onnx"))
        onnx_model = onnx.load(str(tmpdir / "model.onnx"))
        conversion = Conversion.from_onnx_model(onnx_model, model_name=torch_model.__class__.__name__)
        conversion.build(tmpdir)

        directory = Path(save_dir or "conversion").resolve() / conversion.name
        directory.mkdir(exist_ok=True, parents=True)
        shutil.copy(
            str(tmpdir / "build" / conversion.binary_name),
            str(directory / conversion.binary_name)
        )
        ggml_model = GGMLModel(
            str(directory.relative_to(Path.cwd()) / conversion.name).replace("/", "."),
            model_name=torch_model.__class__.__name__,
        )
        # TODO: fix this
        # it seems to sometimes SEGFAULT, when we build the same model twice
        ggml_model.set_weights(torch_model.state_dict())
    return ggml_model
