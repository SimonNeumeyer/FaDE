import math

import numpy
import torch
import torch.nn as nn
import util
from util import Constants, profile_decorator


class DiffNN(nn.ModuleList):
    def __init__(self, settings, graphNNs, stage_index):
        super(DiffNN, self).__init__(graphNNs)
        self.id = util.uuid()
        self.settings = settings
        self.graphNNs = dict([(g.get_id(), g) for g in graphNNs])
        self.stage_index = stage_index
        self.alphas = self._init_alphas()

    def get_id(self):
        return self.id

    def get_graphNNs(self):
        return self.graphNNs

    def get_alpha(self, graphNN_id):
        alpha_list = numpy.array(
            util.get_numeric(self._calculate_alphas(sampling=False))
        )[numpy.array(list(self.graphNNs.keys())) == graphNN_id]
        assert len(alpha_list) == 1
        return alpha_list[0]

    def get_stage_index(self):
        return self.stage_index

    def get_weight_parameters(self):
        return [
            parameter[1]
            for parameter in self.named_parameters()
            if "alphas" not in parameter[0]
        ]

    def _calculate_alphas(self, sampling, train_mode=True):
        if not train_mode:
            return torch.zeros_like(self.alphas).scatter_(
                0, torch.argmax(self.alphas, dim=0), 1.0
            )
        if sampling:
            return nn.functional.gumbel_softmax(self.alphas, hard=True, tau=1)
        else:
            return nn.functional.softmax(self.alphas, dim=0)

    def _init_alphas(self):
        if len(self) == 1:
            return nn.Parameter(torch.ones(len(self)), requires_grad=False)
        if self.settings[Constants.darts_random_init] == 1:
            return nn.Parameter(torch.rand(len(self)), requires_grad=True)
        else:
            return nn.Parameter(torch.ones(len(self)), requires_grad=True)

    def _reduce(self, alphas, tensors):
        if self.settings[Constants.model_operation] == Constants.OPERATION_CONV:
            tensors = [tensor.flatten(2) for tensor in tensors]
        reduced = torch.matmul(torch.stack(tensors, dim=-1), alphas)
        if self.settings[Constants.model_operation] == Constants.OPERATION_CONV:
            width = math.sqrt(reduced.shape[2])
            assert math.isclose(int(width), width)
            reduced = reduced.view(
                reduced.shape[0], reduced.shape[1], int(width), int(width)
            )
        if self.settings[Constants.model_normalize] == 1:
            return reduced / torch.linalg.norm(util.copy_tensor(reduced))
        else:
            return reduced

    def forward(self, x, train_mode):
        alphas = self._calculate_alphas(
            sampling=self.settings[Constants.darts_sampling], train_mode=train_mode
        )
        # if self.alpha_update:
        return self._reduce(alphas, [module(x) for module in self])
        # else:
        #    _, index = torch.max(alphas, dim=0)
        #    return self[index](x)


class GraphNN(nn.Module):
    def __init__(
        self,
        graph,
        settings,
        number_channels=None,
        kernel_size=None,
        stride=None,
        padding=None,
        shared_edges=None,
    ):
        super(GraphNN, self).__init__()
        self.id = util.uuid()
        self.graph = graph
        self.settings = settings
        assert self.settings[Constants.model_operation] in [
            Constants.OPERATION_CONV,
            Constants.OPERATION_LINEAR,
        ]
        if self.settings[Constants.model_operation] == Constants.OPERATION_CONV:
            assert not (
                number_channels is None
                or kernel_size is None
                or stride is None
                or padding is None
            ), "Conv parameters not provided"
            self.number_channels = number_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
        self.edgeNNs = shared_edges
        self._init_edgeNNs()

    def get_id(self):
        return self.id

    def get_graph_id(self):
        return self.graph.get_id()

    def get_graph_id(self):
        return self.graph.get_id()

    def generate_edgeNNs(self, shared=False):
        edgeNNs = nn.ModuleDict()
        edges = self.graph.get_dense_edges() if shared else self.graph.get_edges()
        for edge in edges:
            if self.graph.is_input_edge(*edge):
                edgeNNs[self._stringify_edge(edge)] = nn.Identity()
            else:
                edgeNNs[self._stringify_edge(edge)] = self._get_operation()
        return edgeNNs

    def _init_edgeNNs(self):
        if not self.edgeNNs:
            self.edgeNNs = self.generate_edgeNNs()

    def _get_operation(self):
        if self.settings[Constants.model_operation] == Constants.OPERATION_CONV:
            module = nn.Conv2d(
                self.number_channels,
                self.number_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        else:
            module = nn.Linear(
                self.settings[Constants.mlp_layer_width],
                self.settings[Constants.mlp_layer_width],
            )
        assert self.settings[Constants.model_activation] in [
            Constants.ACTIVATION_RELU
        ], f"Activation {self.activation} not supported"
        activation = nn.ReLU()
        return nn.Sequential(module, activation)

    def _stringify_edge(self, edge):
        return f"edge_{edge[0]}_to_{edge[1]}"

    def _reduce(self, tensors):
        reduced = torch.sum(torch.stack(tensors), dim=0)
        if self.settings[Constants.model_normalize]:
            return reduced / torch.linalg.norm(reduced)
        else:
            return reduced

    def forward(self, x):
        outputs = {self.graph.input_node: x}
        for v in self.graph.get_ordered_nodes():
            inputs = [
                self.edgeNNs[self._stringify_edge((p, v))](outputs[p])
                for p in self.graph.get_predecessors(v)
            ]
            outputs[v] = self._reduce(inputs)
        return outputs[self.graph.output_node]
