import torch
from torch import nn
from shape_extraction import shape_extraction


class test_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(10,20,1)
    def forward(self, x):
        out = self.conv1(x)
        out = torch.nn.functional.relu(input=out,inplace=True)

        return out

if __name__ == "__main__":
    model = test_model()
    input = torch.ones(10,10,10)
    res = shape_extraction(model, input)
    for key in res.keys():
        if key._args:
            print({"api_name": key.name, "input_shape": [res[arg].shape for arg in key._args], "keyword_args": key._kwargs})
