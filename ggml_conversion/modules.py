import abc
from dataclasses import dataclass

import numpy as np
import onnx
import pydantic


def reformat_name(name: str, io_names_map: dict[str, str]) -> str:
    return io_names_map.get(name, name).lstrip("/").replace("/", "_").replace(".", "_")


class Module(pydantic.BaseModel, metaclass=abc.ABCMeta):
    class Config:
        arbitrary_types_allowed = True

    node: onnx.NodeProto
    tensors: dict[str, "Tensor"]
    io_names_map: dict[str, str]

    def reformat_name(self, name: str) -> str:
        return reformat_name(name, self.io_names_map)

    @abc.abstractmethod
    def convert(self) -> str:
        pass


@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)


def broadcast_binary_operation(a: Tensor, b: Tensor) -> tuple[str, str]:
    if a.shape == b.shape:
        return a.name, b.name
    a_shape = a.shape[-2:]
    b_shape = b.shape[-2:]
    if a.ndim != b.ndim:
        if a.ndim > b.ndim:
            return a.name, f"ggml_repeat(ctx, {b.name}, {a.name})"
        return f"ggml_repeat(ctx, {a.name}, {b.name})", b.name
    if any(i > j for i, j in zip(a_shape, b_shape)):
        return a.name, f"ggml_repeat(ctx, {b.name}, {a.name})"
    return f"ggml_repeat(ctx, {a.name}, {b.name})", b.name


def create_load_tensor_data_statement(name: str, data: bytes) -> str:
    return (
        "static const uint8_t {name}_data[] = {{ {data} }};\n"
    )


class Constant(Module):
    def convert(self) -> str:
        data = np.frombuffer(self.node.attribute[0].t.raw_data, dtype=np.float32)
        shape = tuple(self.node.attribute[0].t.dims)
        if shape:
            data = data.reshape(shape)
        ndim = data.ndim
        assert ndim > 0

        if ndim == 1 and len(data) == 1:
            return (
                "struct ggml_tensor *{name} = ggml_new_f32(ctx, {value});"
                .format(
                    name=self.reformat_name(self.node.output[0]),
                    value=data.item(),
                )
            )
        dims = ", ".join(str(dim) for dim in data.shape)
        return (
            "struct ggml_tensor *{name} = ggml_new_tensor_{n}d(ctx, GGML_TYPE_F32, {dims});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                n=ndim,
                dims=dims,
            )
        )


class Add(Module):
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        input_1 = self.reformat_name(self.node.input[1])
        input_0, input_1 = broadcast_binary_operation(
            self.tensors[input_0], self.tensors[input_1]
        )
        return (
            "struct ggml_tensor *{name} = ggml_add(ctx, {input_0}, {input_1});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=input_0,
                input_1=input_1,
            )
        )


class Mul(Module):
    def convert(self) -> str:
        return (
            "struct ggml_tensor *{name} = ggml_mul(ctx, {input_0}, ggml_repeat(ctx, {input_1}, {input_0}));"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=self.reformat_name(self.node.input[0]),
                input_1=self.reformat_name(self.node.input[1]),
            )
        )


class Linear(Module):
    def convert(self) -> str:
        weight = f"model->{self.reformat_name(self.node.input[1])}"
        bias = f"model->{self.reformat_name(self.node.input[2])}"
        return (
            "struct ggml_tensor *{name} = ggml_linear(ctx, {input}, {weight}, {bias});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input=self.reformat_name(self.node.input[0]),
                weight=weight,
                bias=bias,
            )
        )


class ReLU(Module):
    def convert(self) -> str:
        return (
            "struct ggml_tensor *{name} = ggml_relu(ctx, {input});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input=self.reformat_name(self.node.input[0]),
            )
        )


GGML_OPERATORS: dict[str, type[Module]] = {
    "Gemm": Linear,
    "Relu": ReLU,
    "Constant": Constant,
    "Add": Add,
    "Mul": Mul,
}


def broadcastinfo(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    "Get which dimensions are added or repeated when `a` and `b` are broadcast."
    ndim = max(len(a_shape), len(b_shape))

    add_ndims_to_a = ndim - len(a_shape)
    add_ndims_to_b = ndim - len(b_shape)

    a_shape_ = np.array([1] * add_ndims_to_a + list(a_shape))
    b_shape_ = np.array([1] * add_ndims_to_b + list(b_shape))

    if not all((a_shape_ == b_shape_) | (a_shape_ == 1) | (b_shape_ == 1)):
        raise ValueError(f"could not broadcast shapes {a_shape} {b_shape}")

    a_repeatdims = (a_shape_ == 1) & (b_shape_ > 1)  # the repeated dims
    a_repeatdims[:add_ndims_to_a] = True  # the added dims
    a_repeatdims = np.where(a_repeatdims == True)[0]  # indices of axes where True
    a_repeatdims = [int(i) for i in a_repeatdims]

    b_repeatdims = (b_shape_ == 1) & (a_shape_ > 1)
    b_repeatdims[:add_ndims_to_b] = True
    b_repeatdims = np.where(b_repeatdims == True)[0]
    b_repeatdims = [int(i) for i in b_repeatdims]

    return tuple(a_repeatdims), tuple(b_repeatdims)
