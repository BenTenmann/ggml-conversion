import subprocess
import sys
from pathlib import Path

import onnx
import torch
import torch.onnx

from ggml_conversion import core


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 10)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return x


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(exist_ok=True, parents=True)
    torch.onnx.export(
        Net(),
        torch.randn(3, 10),
        str(directory / "model.onnx"),
    )
    model = onnx.load(str(directory / "model.onnx"))
    conversion = core.Conversion.from_onnx_model(model)
    conversion.build(directory)

    subprocess.run([directory / "build" / conversion.name], check=True)


if __name__ == "__main__":
    main()
