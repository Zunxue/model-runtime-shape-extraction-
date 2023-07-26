import torch
import torch.nn as nn
import torch.nn.functional as F


def shape_extraction(model: nn.Module, input):
    graph_model = torch.fx.symbolic_trace(model)
    graph_model = torch.fx.Interpreter(graph_model, garbage_collect_values=False)
    graph_model.run(input)
    return graph_model.env
    