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

    @property
    def ggml_shape(self) -> tuple[int, ...]:
        # ggml uses row-major order
        return self.shape[::-1]


def broadcast_binary_operation(a: Tensor, b: Tensor) -> tuple[str, str]:
    if a.shape == b.shape:
        return a.name, b.name
    a_shape = a.ggml_shape
    b_shape = b.ggml_shape
    if a.ndim != b.ndim:
        dim_diff = abs(a.ndim - b.ndim)
        min_shape = b_shape if a.ndim > b.ndim else a_shape
        dims = ', '.join(str(dim) for dim in min_shape + (1,) * dim_diff)
        if a.ndim > b.ndim:
            # TODO: check if this is correct
            return a.name, f"ggml_repeat(ctx, ggml_reshape_{a.ndim}d(ctx, {b.name}, {dims}), {a.name})"
        return f"ggml_repeat(ctx, ggml_reshape_{b.ndim}d(ctx, {a.name}, {dims}), {b.name})", b.name
    if any(i > j for i, j in zip(a_shape, b_shape)):
        return a.name, f"ggml_repeat(ctx, {b.name}, {a.name})"
    return f"ggml_repeat(ctx, {a.name}, {b.name})", b.name


def create_load_tensor_data_statement(name: str, data: bytes) -> str:
    return (
        f"""
        std::vector<unsigned char> {name}_data = {{
            {", ".join(str(b) for b in data)}
        }};
        """
        + "std::memcpy({name}->data, {name}_data.data(), ggml_nbytes({name}));"
        .format(
            name=name,
            data=data,
        )
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
        dims = ", ".join(str(dim) for dim in data.shape[::-1])
        return (
            "struct ggml_tensor *{name} = ggml_new_tensor_{n}d(ctx, GGML_TYPE_F32, {dims});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                n=ndim,
                dims=dims,
            )
            + create_load_tensor_data_statement(
                self.reformat_name(self.node.output[0]),
                data.tobytes(),
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


class Sub(Module):
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        input_1 = self.reformat_name(self.node.input[1])
        input_0, input_1 = broadcast_binary_operation(
            self.tensors[input_0], self.tensors[input_1]
        )
        return (
            "struct ggml_tensor *{name} = ggml_sub(ctx, {input_0}, {input_1});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=input_0,
                input_1=input_1,
            )
        )


class Mul(Module):
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        input_1 = self.reformat_name(self.node.input[1])
        input_0, input_1 = broadcast_binary_operation(
            self.tensors[input_0], self.tensors[input_1]
        )
        return (
            "struct ggml_tensor *{name} = ggml_mul(ctx, {input_0}, {input_1});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=input_0,
                input_1=input_1,
            )
        )


class Div(Module):
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        input_1 = self.reformat_name(self.node.input[1])
        input_0, input_1 = broadcast_binary_operation(
            self.tensors[input_0], self.tensors[input_1]
        )
        return (
            "struct ggml_tensor *{name} = ggml_div(ctx, {input_0}, {input_1});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=input_0,
                input_1=input_1,
            )
        )


class Pow(Module):
    # TODO: check if this is correct
    # not sure if there is `ggml_pow` function
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        input_1 = self.reformat_name(self.node.input[1])
        input_0, input_1 = broadcast_binary_operation(
            self.tensors[input_0], self.tensors[input_1]
        )
        return (
            "struct ggml_tensor *{name} = ggml_pow(ctx, {input_0}, {input_1});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=input_0,
                input_1=input_1,
            )
        )


class Sqrt(Module):
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        return (
            "struct ggml_tensor *{name} = ggml_sqrt(ctx, {input_0});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=input_0,
            )
        )


class MatMul(Module):
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        input_1 = self.reformat_name(self.node.input[1])
        return (
            # for now, we need to ensure that the first input is contiguous
            # otherwise, GGML will crash
            "struct ggml_tensor *{name} = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, {input_0}, {input_1})));"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=input_0,
                input_1=input_1,
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


class Softmax(Module):
    def convert(self) -> str:
        return (
            "struct ggml_tensor *{name} = ggml_soft_max(ctx, {input});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input=self.reformat_name(self.node.input[0]),
            )
        )


class Tanh(Module):
    def convert(self) -> str:
        return (
            "struct ggml_tensor *{name} = ggml_tanh(ctx, {input});"
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
    "Sub": Sub,
    "Mul": Mul,
    "Div": Div,
    # "Pow": Pow,
    "Sqrt": Sqrt,
    "Softmax": Softmax,
    "MatMul": MatMul,
}
