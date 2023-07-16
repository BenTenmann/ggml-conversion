import abc
import enum
from dataclasses import dataclass
from typing import Final

import numpy as np
import onnx
import pydantic

ONNX_DATA_TYPE_MAP: Final[dict[int, np.dtype]] = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    4: np.uint16,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    8: str,       # String
    9: object,    # Object
    10: np.bool_,     # Boolean
    11: np.float16,
    12: np.double,
    13: np.uint32,
    14: np.uint64,
    15: np.complex64,
    16: np.complex128,
    17: np.float16,  # Float16Alt
}


def reformat_name(name: str, io_names_map: dict[str, str]) -> str:
    return io_names_map.get(name, name).lstrip("/").replace("/", "_").replace(".", "_")


class Module(pydantic.BaseModel, metaclass=abc.ABCMeta):
    class Config:
        arbitrary_types_allowed = True

    node: onnx.NodeProto
    graph: onnx.GraphProto
    tensors: dict[str, "Tensor"]
    io_names_map: dict[str, str]

    def reformat_name(self, name: str) -> str:
        return reformat_name(name, self.io_names_map)

    @abc.abstractmethod
    def convert(self) -> str:
        pass

    def get_node(self, node_name: str) -> onnx.NodeProto:
        for node in self.graph.node:
            if node.name == node_name:
                return node
        raise ValueError(f"Node {node_name} not found in graph")


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

    @property
    def n_elements(self) -> int:
        return int(np.prod(self.shape))


class BroadcastTensor(enum.Enum):
    A = "a"
    B = "b"
    NONE = "none"


@dataclass
class BroadcastOp:
    a: str
    b: str
    broadcast_tensor: BroadcastTensor

    @property
    def broadcast(self) -> str | None:
        if self.broadcast_tensor == BroadcastTensor.A:
            return self.a
        if self.broadcast_tensor == BroadcastTensor.B:
            return self.b
        return None


def broadcast_binary_operation(a: Tensor, b: Tensor) -> BroadcastOp:
    if a.shape == b.shape:
        return BroadcastOp(a.name, b.name, BroadcastTensor.NONE)
    a_shape = a.ggml_shape
    b_shape = b.ggml_shape
    if a.ndim != b.ndim:
        dim_diff = abs(a.ndim - b.ndim)
        min_shape = b_shape if a.ndim > b.ndim else a_shape
        dims = ', '.join(str(dim) for dim in min_shape + (1,) * dim_diff)
        if a.ndim > b.ndim:
            # TODO: check if this is correct
            return BroadcastOp(
                a.name,
                f"ggml_repeat(ctx, ggml_reshape_{a.ndim}d(ctx, {b.name}, {dims}), {a.name})",
                BroadcastTensor.B
            )
        return BroadcastOp(
            f"ggml_repeat(ctx, ggml_reshape_{b.ndim}d(ctx, {a.name}, {dims}), {b.name})",
            b.name,
            BroadcastTensor.A
        )
    if any(i > j for i, j in zip(a_shape, b_shape)):
        return BroadcastOp(
            a.name,
            f"ggml_repeat(ctx, {b.name}, {a.name})",
            BroadcastTensor.B
        )
    return BroadcastOp(
        f"ggml_repeat(ctx, {a.name}, {b.name})",
        b.name,
        BroadcastTensor.A
    )


def create_load_tensor_data_statement(name: str, data: bytes) -> str:
    return (
        f"""
        constexpr unsigned char {name}_data[] = {{
            {", ".join(str(b) for b in data)}
        }};
        """
        + "std::memcpy({name}->data, {name}_data, ggml_nbytes({name}));"
        .format(
            name=name,
            data=data,
        )
    )


def decode_tensor_data(node: onnx.NodeProto) -> np.ndarray:
    return np.frombuffer(node.attribute[0].t.raw_data, dtype=ONNX_DATA_TYPE_MAP[node.attribute[0].t.data_type])


class Constant(Module):
    def convert(self) -> str:
        data = decode_tensor_data(self.node)
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


class ConstantOfShape(Module):
    def convert(self) -> str:
        data = decode_tensor_data(self.node)
        assert len(data) == 1, "Only scalar values are supported"

        constant_node = self.get_node(self.node.input[0].replace("_output_0", ""))
        shape: tuple[int, ...] = tuple(decode_tensor_data(constant_node))[::-1]
        return (
            "struct ggml_tensor *{name} = ggml_new_tensor_{n}d(ctx, GGML_TYPE_F32, {dims});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                n=len(shape),
                dims=", ".join(str(dim) for dim in shape),
            )
            + "ggml_set_f32({name}, {value});".format(
                    name=self.reformat_name(self.node.output[0]),
                    value=data.item(),
            )
        )


class Add(Module):
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        input_1 = self.reformat_name(self.node.input[1])
        broadcast = broadcast_binary_operation(
            self.tensors[input_0], self.tensors[input_1]
        )
        return (
            "struct ggml_tensor *{name} = ggml_add(ctx, {input_0}, {input_1});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=broadcast.a,
                input_1=broadcast.b,
            )
        )


class Sub(Module):
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        input_1 = self.reformat_name(self.node.input[1])
        broadcast = broadcast_binary_operation(
            self.tensors[input_0], self.tensors[input_1]
        )
        return (
            "struct ggml_tensor *{name} = ggml_sub(ctx, {input_0}, {input_1});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=broadcast.a,
                input_1=broadcast.b,
            )
        )


class Mul(Module):
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        input_1 = self.reformat_name(self.node.input[1])
        broadcast = broadcast_binary_operation(
            self.tensors[input_0], self.tensors[input_1]
        )
        return (
            "struct ggml_tensor *{name} = ggml_mul(ctx, {input_0}, {input_1});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=broadcast.a,
                input_1=broadcast.b,
            )
        )


class Div(Module):
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        input_1 = self.reformat_name(self.node.input[1])
        broadcast = broadcast_binary_operation(
            self.tensors[input_0], self.tensors[input_1]
        )
        return (
            "struct ggml_tensor *{name} = ggml_div(ctx, {input_0}, {input_1});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=broadcast.a,
                input_1=broadcast.b,
            )
        )


class Pow(Module):
    # TODO: check if this is correct
    # not sure if there is `ggml_pow` function
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        input_1 = self.reformat_name(self.node.input[1])
        broadcast = broadcast_binary_operation(
            self.tensors[input_0], self.tensors[input_1]
        )
        return (
            "struct ggml_tensor *{name} = ggml_pow(ctx, {input_0}, {input_1});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=broadcast.a,
                input_1=broadcast.b,
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


class Log(Module):
    def convert(self) -> str:
        return (
            "struct ggml_tensor *{name} = ggml_log(ctx, {input});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input=self.reformat_name(self.node.input[0]),
            )
        )


class Transpose(Module):
    def convert(self) -> str:
        return (
            "struct ggml_tensor *{name} = ggml_cont(ctx, ggml_transpose(ctx, {input}));"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input=self.reformat_name(self.node.input[0]),
            )
        )


class Erf(Module):
    def convert(self) -> str:
        return (
            "struct ggml_tensor *{name} = ggml_erf(ctx, {input});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input=self.reformat_name(self.node.input[0]),
            )
        )


class Reshape(Module):
    def convert(self) -> str:
        input_name = self.reformat_name(self.node.input[0])

        shape_name = self.reformat_name(self.node.input[1])
        shape_tensor = self.tensors[shape_name]

        constant_node = self.get_node(self.node.input[1].replace("_output_0", ""))
        reshape_tuple: tuple[int, ...] = tuple(decode_tensor_data(constant_node))[::-1]
        if any(i == -1 for i in reshape_tuple):
            input_tensor = self.tensors[input_name]
            # taking the negative, as there is exactly one -1 in reshape_tuple
            inferred_dim = - int(input_tensor.n_elements / np.prod(reshape_tuple))
            reshape_tuple = tuple(
                inferred_dim if i == -1 else i for i in reshape_tuple
            )
        return (
            "struct ggml_tensor *{name} = ggml_reshape_{n}d(ctx, {input}, {shape});"
            .format(
                n=shape_tensor.ndim,
                name=self.reformat_name(self.node.output[0]),
                input=input_name,
                shape=", ".join(str(i) for i in reshape_tuple),
            )
        )


class ReduceMean(Module):
    def convert(self) -> str:
        # TODO: support other axes
        # currently this only does the mean over the rows
        return (
            "struct ggml_tensor *{name} = ggml_mean(ctx, {input});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input=self.reformat_name(self.node.input[0]),
            )
        )


class Equal(Module):
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        input_1 = self.reformat_name(self.node.input[1])
        broadcast = broadcast_binary_operation(
            self.tensors[input_0], self.tensors[input_1]
        )
        return (
            "struct ggml_tensor *{name} = ggml_equal(ctx, {input_0}, {input_1});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=broadcast.a,
                input_1=broadcast.b,
            )
        )


class Where(Module):
    def convert(self) -> str:
        input_0 = self.reformat_name(self.node.input[0])
        input_1 = self.reformat_name(self.node.input[1])
        input_2 = self.reformat_name(self.node.input[2])

        broadcast = broadcast_binary_operation(
            self.tensors[input_0], self.tensors[input_1]
        )
        broadcast2 = broadcast_binary_operation(
            self.tensors[broadcast.broadcast or broadcast.a],
            self.tensors[input_2]
        )
        return (
            "struct ggml_tensor *{name} = ggml_where(ctx, {condition}, {x}, {y});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                condition=broadcast.a,
                x=broadcast.b,
                y=broadcast2.b,
            )
        )


class Expand(Module):
    # TODO: fix this
    # real issue: GGML does not support runtime reshaping
    # as a potential solution, we could recurse through the graph and find all the constant nodes and then compute the
    # shape tensors at compile time
    # this would be a lot of work, and more importantly, it would break if the shape tensor is not a constant
    def convert(self) -> str:
        raise NotImplementedError


GGML_OPERATORS: dict[str, type[Module]] = {
    "Gemm": Linear,
    "Relu": ReLU,
    "Constant": Constant,
    "ConstantOfShape": ConstantOfShape,
    "Add": Add,
    "Sub": Sub,
    "Mul": Mul,
    "Div": Div,
    "Pow": Pow,
    "Sqrt": Sqrt,
    "Softmax": Softmax,
    "MatMul": MatMul,
    "Tanh": Tanh,
    "Log": Log,
    "Transpose": Transpose,
    "Erf": Erf,
    "Reshape": Reshape,
    # "ReduceMean": ReduceMean,
    "Equal": Equal,
    "Where": Where,
}
