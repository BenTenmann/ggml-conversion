import importlib
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
PYBIND11_REPO: Final[str] = "https://github.com/pybind/pybind11.git"


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
            dims=", ".join(str(d) for d in dims)
        )
        for name, dims in zip(weights, dimension)
    )


def _generate_array_type(value: onnx.ValueInfoProto) -> str:
    arr_template = "std::array<{type}, {size}>"
    arr = arr_template
    for d in list(value.type.tensor_type.shape.dim)[:-1]:
        arr = arr.format(
            type=arr_template,
            size=d.dim_value
        )
    arr = arr.format(
        type="float",
        size=list(value.type.tensor_type.shape.dim)[-1].dim_value
    )
    return arr


def generate_input_type(model: onnx.ModelProto) -> str:
    inp = model.graph.input[0]
    return _generate_array_type(inp)


def generate_output_type(model: onnx.ModelProto) -> str:
    out = model.graph.output[0]
    return _generate_array_type(out)


def _generate_nested_for_loop_with_value_setting(
    value: onnx.ValueInfoProto,
    input_name: str,
    output_name: str,
) -> str:
    for_loop_template = "for (int {idx} = 0; {idx} < {size}; {idx}++) {body}"
    for_loop = for_loop_template
    indices = ("i", "j", "k", "l")[:len(list(value.type.tensor_type.shape.dim))]
    for idx, d in zip(indices, list(value.type.tensor_type.shape.dim)[:-1]):
        for_loop = for_loop.format(
            size=d.dim_value,
            body=for_loop_template,
            idx=idx,
        )
    for_loop = for_loop.format(
        size=list(value.type.tensor_type.shape.dim)[-1].dim_value,
        body="ggml_set_f32_{ndim}d({output_name}, {indices}, {input_name}{idx});".format(
            output_name=output_name,
            input_name=input_name,
            ndim=len(indices),
            indices=", ".join(map(str, indices)),
            idx="".join(f"[{idx}]" for idx in indices)
        ),
        idx=indices[-1],
    )
    return for_loop


def generate_set_input(model: onnx.ModelProto) -> str:
    value = model.graph.input[0]
    return "\n    ".join(
        [
            "struct ggml_tensor *{output_name} = ggml_new_tensor_{ndim}d(ctx, GGML_TYPE_F32, {shape});".format(
                output_name="output",
                ndim=len(list(value.type.tensor_type.shape.dim)),
                shape=", ".join(str(d.dim_value) for d in value.type.tensor_type.shape.dim),
            ),
            _generate_nested_for_loop_with_value_setting(
                value=value,
                input_name="input",
                output_name="output",
            ),
            "return output;"
        ]
    )


def generate_get_output(model: onnx.ModelProto) -> str:
    value = model.graph.output[0]
    return "\n    ".join(
        [
            "{output_type} {input_name} = {{}};".format(
                input_name="input",
                output_type=generate_input_type(model),
            ),
            _generate_nested_for_loop_with_value_setting(
                value=value,
                input_name="input",
                output_name="output",
            ),
            "return input;"
        ]
    )


def generate_weights_type(value: onnx.ValueInfoProto) -> str:
    return "int"


def generate_set_weights(model: onnx.ModelProto) -> str:
    return "\n    ".join(
        _generate_nested_for_loop_with_value_setting(
            value=init,
            input_name=modules.reformat_name(init.name, create_io_name_map(model)),
            output_name="model->{}".format(
                modules.reformat_name(init.name, create_io_name_map(model))
            ),
        )
        for init in model.graph.initializer
    )


def get_input_shape(model: onnx.ModelProto) -> tuple[int, int]:
    return (
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_value,
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    )


class GGMLModel:
    def __init__(self, model_path: str, model_weights: dict[str, torch.Tensor]):
        self.model = importlib.import_module(model_path).Model()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            self.model.forward(x.tolist()), dtype=x.dtype, device=x.device
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
        git.Repo.clone_from(PYBIND11_REPO, path / "pybind11")
        (path / "main.cpp").write_text(self.main)
        (path / "CMakeLists.txt").write_text(self.cmakelists)
        build_dir = path / "build"
        build_dir.mkdir(exist_ok=True, parents=True)
        subprocess.run(["cmake", ".."], cwd=build_dir)
        subprocess.run(["make"], cwd=build_dir)

    @classmethod
    def from_onnx_model(cls, model: onnx.ModelProto) -> "Conversion":
        name = modules.reformat_name(model.graph.name, create_io_name_map(model))
        return cls(
            name=name,
            main=templates.MAIN.format(
                model_struct=generate_model_struct(model),
                forward=generate_forward(model),
                model_init=generate_model_init(model),
                input_type=generate_input_type(model),
                output_type=generate_output_type(model),
                set_input=generate_set_input(model),
                get_output=generate_get_output(model),
                mem_size="1024 * 1024 * 1024",
                weights_type=generate_weights_type(model),
                set_weights=generate_set_weights(model),
                model_name=name,

            ),
            cmakelists=templates.CMAKELISTS.format(
                model_name=name,
            ),
        )


def run_ggml_converter(
    model: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
) -> GGMLModel:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        torch.onnx.export(model, args, str(tmpdir / "model.onnx"))
        model = onnx.load(str(tmpdir / "model.onnx"))
        conversion = Conversion.from_onnx_model(model)
        conversion.build(tmpdir)
