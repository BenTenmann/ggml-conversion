import subprocess
import tempfile
from pathlib import Path
from typing import Final, Literal

import git
import onnx
import pydantic
import torch

from ggml_conversion import modules, templates

GGML_REPO: Final[str] = "https://github.com/ggerganov/ggml.git"


def create_io_name_map(model: onnx.ModelProto) -> dict[str, Literal["input", "output"]]:
    # TODO: handle multiple inputs and outputs
    # input and output names are often not allowed in C
    return {
        model.graph.input[0].name: "input",
        model.graph.output[0].name: "output",
    }


def generate_forward(model: onnx.ModelProto) -> str:
    io_name_map = create_io_name_map(model)
    mods: list[str] = []
    for node in model.graph.node:
        if node.op_type not in modules.GGML_OPERATORS:
            raise NotImplementedError(f"Operator {node.op_type} is not supported")
        mods.append(modules.GGML_OPERATORS[node.op_type](node=node, io_names_map=io_name_map).convert())
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
            dims=", ".join(str(d) for d in dims)
        )
        for name, dims in zip(weights, dimension)
    )


def generate_set_input(model: onnx.ModelProto) -> str:
    return "\n    ".join(
        "ggml_set_f32({name}, 2.0);".format(
            name=name
        )
        for name in ["input"] +
        [
            "model." + modules.reformat_name(init.name, create_io_name_map(model))
            for init in model.graph.initializer
        ]
    )


def get_input_shape(model: onnx.ModelProto) -> tuple[int, int]:
    return (
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_value,
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    )


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
        (path / "main.cpp").write_text(self.main)
        (path / "CMakeLists.txt").write_text(self.cmakelists)
        build_dir = path / "build"
        build_dir.mkdir(exist_ok=True, parents=True)
        subprocess.run(["cmake", ".."], cwd=build_dir)
        subprocess.run(["make"], cwd=build_dir)

    @classmethod
    def from_onnx_model(cls, model: onnx.ModelProto) -> "Conversion":
        dim_0, dim_1 = get_input_shape(model)
        name = modules.reformat_name(model.graph.name, create_io_name_map(model))
        return cls(
            name=name,
            main=templates.MAIN.format(
                model_struct=generate_model_struct(model),
                forward=generate_forward(model),
                model_init=generate_model_init(model),
                set_input=generate_set_input(model),
                mem_size="1024 * 1024 * 1024",
                dim_1=dim_0,
                dim_0=dim_1,
            ),
            cmakelists=templates.CMAKELISTS.format(
                model_name=name,
            ),
        )


def run_ggml_converter(
    model: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        torch.onnx.export(model, args, str(tmpdir / "model.onnx"))
