import collections
import importlib
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

from ggml_conversion import modules, templates

GGML_REPO: Final[str] = "https://github.com/ggerganov/ggml.git"
PYBIND11_REPO: Final[str] = "https://github.com/pybind/pybind11.git"
FLOAT_BYTES: Final[int] = 4

# not sure if this is correct (taken from https://github.com/ggerganov/ggml/blob/master/examples/gpt-2/main.cpp#L194)
OBJECT_OVERHEAD: Final[int] = 512


def create_io_name_map(model: onnx.ModelProto) -> dict[str, Literal["input", "output"]]:
    # TODO: handle multiple inputs and outputs
    # input and output names are often not allowed in C
    return {
        model.graph.input[0].name: "input",
        model.graph.output[0].name: "output",
    }


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
        self.model = getattr(module, model_name)()
        self.model_name = model_name

    def set_weights(self, weights: collections.OrderedDict[str, torch.Tensor]) -> None:
        self.model.set_weights({k: v.ravel().tolist() for k, v in weights.items()})

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            self.model.forward(x.ravel().tolist()), dtype=x.dtype, device=x.device
        ).view(*self.model.output_shape())


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
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        git.Repo.clone_from(GGML_REPO, path / "ggml")
        git.Repo.clone_from(PYBIND11_REPO, path / "pybind11")
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
                output_ndim=len(model.graph.output[0].type.tensor_type.shape.dim),
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


def run_ggml_converter(
    torch_model: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    save_dir: Path | str | None = None,
) -> GGMLModel:
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
