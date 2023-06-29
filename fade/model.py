import functools
import math
import random

import torch.nn as nn
import torch.nn.functional as F
from diffNN import *
from util import Constants


class AbstractModel(nn.Module):
    def __init__(
        self,
        settings,
        graphs,
        dataset_number_channels,
        dataset_number_features,
        dataset_number_classes,
    ):
        super(AbstractModel, self).__init__()
        self.settings = settings
        self.graphs = graphs
        self.dataset_number_channels = dataset_number_channels
        self.dataset_number_features = dataset_number_features
        self.dataset_number_classes = dataset_number_classes
        self.stages = []

    def _interface(self):
        """execute AFTER creating stages"""
        self.stage_dict = dict([(s.get_id(), s) for s in self.stages])
        self.key_stage_id = "key_stage_id"
        self.key_module = "key_module"
        # info: {uuid: {"key_stage_id": uuid, "key_module": graphNN object}, uuid: ...}
        self.graphNNs = functools.reduce(
            lambda a, b: {**a, **b},
            [
                {
                    g: {
                        self.key_stage_id: s.get_id(),
                        self.key_module: s.get_graphNNs()[g],
                    }
                }
                for s in self.stages
                for g in s.get_graphNNs()
            ],
        )

    def get_number_stages(self):
        return self.settings[Constants.model_number_stages]

    def get_graphNN_ids(self):
        """Return graphNN ids"""
        return self.graphNNs.keys()

    def get_stage_index(self, graphNN_id):
        """Returns stage index (0,1,...) for graphNN id"""
        assert graphNN_id in self.graphNNs
        return self.stage_dict[
            self.graphNNs[graphNN_id][self.key_stage_id]
        ].get_stage_index()

    def get_graphNN(self, graphNN_id):
        """Return graphNN object for id"""
        assert graphNN_id in self.graphNNs
        return self.graphNNs[graphNN_id][self.key_module]

    def get_parameters(self, alphas):
        """alphas ? return all alpha parameters : return all none alpha parameters"""
        params = []
        for name, param in self.named_parameters():
            if (
                self.is_alpha_parameter(name)
                and alphas
                or not self.is_alpha_parameter(name)
                and not alphas
            ):
                params.append(param)
        return params

    def is_alpha_parameter(self, name):
        """Caution: cross reference to attribute naming of diffNN class (no constant possible)"""
        return "alphas" in name

    def get_alpha(self, graphNN_id):
        """Fetches alpha parameter for graphNN id from corresponding stage/diffNN object"""
        assert graphNN_id in self.graphNNs
        return self.stage_dict[self.graphNNs[graphNN_id][self.key_stage_id]].get_alpha(
            graphNN_id
        )

    def set_alpha_update(
        self, alphaUpdate
    ):  # Deprecated, proper DARTS implementation should not need it
        for stage in self.stages:
            stage.set_alpha_update(alphaUpdate)


class TestModel(AbstractModel):
    pass


class MLP(AbstractModel):  # Multi Layer Perceptron
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.features = (
            kwargs["dataset_number_channels"] * kwargs["dataset_number_features"]
        )
        assert self.settings[Constants.model_activation] in [
            Constants.ACTIVATION_RELU
        ], f"Activation {self.activation} not supported"
        self.linear_0 = nn.Sequential(
            nn.Linear(self.features, self.settings[Constants.mlp_layer_width]),
            nn.ReLU(),
        )
        self.linear_1 = nn.Sequential(
            nn.Linear(
                self.settings[Constants.mlp_layer_width],
                kwargs["dataset_number_classes"],
            )
        )
        self._create_stages()
        self._interface()

    def _create_stages(self):
        self.stages = nn.ModuleList()
        for i in range(self.settings[Constants.model_number_stages]):
            # if self.settings[Constants.stage_shared_weights]:
            #    graphNNs = GraphNN.generate_graphNNs_shared_weights(graphs, self.settings, additional_parameters={"layerWidth": self.layer_width})
            # else:
            assert self.settings[Constants.stage_shared_weights] == 0
            graphNNs = [GraphNN(graph, self.settings) for graph in self.graphs]
            diffNN = DiffNN(settings=self.settings, graphNNs=graphNNs, stage_index=i)
            self.stages.append(diffNN)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.linear_0(x)
        for cell in self.stages:
            x = cell(x, self.training)
        x = self.linear_1(x)
        return x


class ConvolutionalModel(AbstractModel):
    def __init__(self, **kwargs):
        super(ConvolutionalModel, self).__init__(**kwargs)
        self.number_features = round(math.sqrt(self.dataset_number_features))
        self.final_number_channels = self.settings[Constants.conv_final_number_channels]
        self.shared_weights = self.settings[Constants.stage_shared_weights]
        self.number_stages = self.settings[Constants.model_number_stages]
        self.pooling_type = self.settings[Constants.conv_pooling]
        self.kernel_size = self.settings[Constants.conv_kernel_size]
        self._create_stages()
        self._interface()

    def _create_stages(self):
        assert math.isclose(
            round(self.number_features), self.number_features
        ), f"Number features {self.dataset_number_features} no square"
        assert self.pooling_type in [
            Constants.POOLING_MAX,
            Constants.POOLING_AVG,
        ], f"Pooling {self.pooling_operation} not supported"
        assert self.kernel_size in [
            3,
            5,
            7,
        ], f"Kernel size {self.kernel_size} not supported"
        assert (self.final_number_channels) >= 1
        self.pooling_kernel_size = 2
        self.stride = 1
        base, self.padding, channel_counts = self._calculate_pooling(
            self.final_number_channels, self.number_stages, self.number_features
        )
        self.convolutional_input_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.poolings = nn.ModuleList()
        self._create_convolutional_layers(channel_counts)
        self._create_pooling_layers(base)
        if self.number_stages == 1:
            self.linear_1 = nn.Linear(
                self.final_number_channels * 8 * 8, self.dataset_number_classes
            )
        else:
            self.linear_1 = nn.Linear(
                self.final_number_channels * 2 * 2, self.dataset_number_classes
            )

    def _create_convolutional_layers(self, channel_counts):
        conv_parameters = {
            "kernel_size": self.kernel_size,
            "padding": int((self.kernel_size - 1) / 2),
            "stride": self.stride,
        }
        channels = [int(c) for c in [self.dataset_number_channels] + channel_counts]
        for i in range(self.number_stages):
            # stage_input
            convolutional_input_layer = nn.Conv2d(
                channels[i],
                channels[i + 1],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=int((self.kernel_size - 1) / 2),
            )
            self.convolutional_input_layers.append(convolutional_input_layer)
            # stage
            conv_parameters["number_channels"] = channels[i + 1]
            # if self.settings[Constants.stage_shared_weights]:
            #    graphNNs = GraphNN.generate_graphNNs_shared_weights(graphs, self.settings, additional_parameters=conv_parameters)
            # else:
            graphNNs = [
                GraphNN(graph=graph, settings=self.settings, **conv_parameters)
                for graph in self.graphs
            ]
            diffNN = DiffNN(settings=self.settings, graphNNs=graphNNs, stage_index=i)
            self.stages.append(diffNN)

    def _create_pooling_layers(self, base):
        for i in range(self.number_stages):
            if base[i] == 1:
                pooling_layer = nn.Identity()
            elif self.pooling_type == Constants.POOLING_AVG:
                pooling_layer = torch.nn.AvgPool2d(
                    base[i], stride=base[i], padding=self.padding[i]
                )
            else:
                pooling_layer = torch.nn.MaxPool2d(
                    base[i], stride=base[i], padding=self.padding[i]
                )
            self.poolings.append(pooling_layer)

    def _calculate_pooling(self, final_channel_count, number_stages, number_features):
        # kernel
        assert number_stages > 0
        if number_stages < 3:
            base = [4, 4][:number_stages]
        elif number_stages == 3:
            base = [4, 2, 2]
        else:
            indices = random.sample(range(number_stages - 2), 2)
            base = (
                [2] + [2 if i in indices else 1 for i in range(number_stages - 2)] + [2]
            )
        # padding
        assert number_features in [
            28,
            32,
        ], "Implementation only for MNIST and CIFAR10 with corresponding features sizes 28x28 resp. 32x32"
        padding = [0 for i in range(number_stages)]
        if number_features == 28:
            if number_stages == 1:
                padding[0] = 2
            else:
                padding = [
                    1
                    if functools.reduce(lambda x, y: x * y, base[:i], 1) == 4
                    and base[i] > 1
                    else 0
                    for i in range(number_stages)
                ]

        # channel
        channel_counts = [
            final_channel_count
            if i == 0
            else final_channel_count
            / functools.reduce(lambda x, y: x * y, base[-i:], 1)
            for i in range(number_stages)
        ]
        channel_counts.reverse()
        assert all(
            [math.isclose(round(c), c) for c in channel_counts]
        ), "Channel count not even enough"
        return (base, padding, channel_counts)

    def forward(self, x):
        for i in range(self.number_stages):
            x = self.convolutional_input_layers[i](x)
            x = self.stages[i](x, self.training)
            x = self.poolings[i](x)
        x = x.view(x.size()[0], -1)
        x = self.linear_1(x)
        return x
