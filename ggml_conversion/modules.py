import abc

import numpy as np
import onnx
import pydantic


def reformat_name(name: str, io_names_map: dict[str, str]) -> str:
    return io_names_map.get(name, name).lstrip("/").replace("/", "_").replace(".", "_")


class Module(pydantic.BaseModel, metaclass=abc.ABCMeta):
    class Config:
        arbitrary_types_allowed = True

    node: onnx.NodeProto
    io_names_map: dict[str, str]

    def reformat_name(self, name: str) -> str:
        return reformat_name(name, self.io_names_map)

    @abc.abstractmethod
    def convert(self) -> str:
        pass


def requires_broadcast(node: onnx.NodeProto) -> bool:
    pass


def broadcast():
    pass


class Constant(Module):
    def convert(self) -> str:
        data = np.frombuffer(self.node.attribute[0].t.raw_data, dtype=np.float32)
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
        raise NotImplementedError

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
        return (
            "struct ggml_tensor *{name} = ggml_add(ctx, {input_0}, {input_1});"
            .format(
                name=self.reformat_name(self.node.output[0]),
                input_0=self.reformat_name(self.node.input[0]),
                input_1=self.reformat_name(self.node.input[1]),
            )
        )


class Mul(Module):
    def convert(self) -> str:
        return (
            "struct ggml_tensor *{name} = ggml_mul(ctx, {input_0}, {input_1});"
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
