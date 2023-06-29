import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy
import torch
from torch.utils.tensorboard import SummaryWriter

from .util import *


def visualize_flag(func):
    def wrapper_visualize_flag(self, *args, **kwargs):
        if self.visualize:
            func(self, *args, **kwargs)

    return wrapper_visualize_flag


def convert_tensor(tensor):
    return tensor.cpu()


class Visualization:
    def __init__(self, flag=True):
        self.visualize = flag
        if self.visualize:
            self.writer = SummaryWriter("../tensorboard/")

    def register_diffNN(self, diffNN):
        self.diffNN_registry.append(diffNN)

    def _visualize_graph(self, networkx_graph):
        nx.draw_networkx(networkx_graph)
        return plt.gcf()

    def _write_settings(self, settingsString):
        self.writer.add_text("Settings", settingsString)

    def save_alphas(self):
        for diffNN in self.diffNN_registry:
            alphas = diffNN.get_alphas()
            path_save = os.path.join(
                self.result_path, diffNN.get_name(), f"{Constants.ALPHA}.pt"
            )
            torch.save(alphas, path_save)

    @visualize_flag
    def plot_model(self, model, sample=None):
        self.writer.add_graph(model, sample)

    @visualize_flag
    def plot_graphs(self, graphs):
        for i, g in enumerate(graphs):
            fig = self._visualize_graph(g)
            self.writer.add_figure(figure=fig, tag="_".join(["Graphs"]), global_step=i)

    @visualize_flag
    def training_loss(self, epoch, counter, loss):
        """
        Args:
            loss: batch loss

        Collects loss information for learning rate visualization.
        """
        self.running_loss = ((counter - 1) * self.running_loss + loss) / counter

    @visualize_flag
    def visualize_alphas(self):
        for diffNN in self.diffNN_registry:
            self.writer.add_text(
                "_".join([diffNN.get_name(), Constants.ALPHA]), str(diffNN.get_alphas())
            )

    @visualize_flag
    def evaluation_loss(self, epoch, evaluation_loss, accuracy):
        """
        Args:
            alphas: dictionary with 'title' - 'weights' as key - value structure
            optimizationSettings: optimizationSettings object

        Visualizes losses and given alphas to TensorBoard.
        """
        # Training loss
        self.writer.add_scalar(
            tag=self.title_training_loss,
            scalar_value=self.running_loss,
            global_step=epoch,
        )
        self.running_loss = 0
        # Evaluation loss
        self.writer.add_scalar(
            tag=self.title_evaluation_loss,
            scalar_value=evaluation_loss,
            global_step=epoch,
        )
        # Accuracy
        self.writer.add_scalar(
            tag=self.title_accuracy, scalar_value=accuracy, global_step=epoch
        )
        self.accuracy = accuracy

        # alphas
        for diffNN in self.diffNN_registry:
            self.writer.add_figure(
                figure=self._get_alpha_plot(alpha=diffNN.get_alphas()),
                tag="_".join([diffNN.get_name(), Constants.ALPHA]),
                global_step=epoch,
            )

    @visualize_flag
    def close(self):
        self.writer.flush()
        self.writer.close()

    def _get_alpha_plot(self, alpha):
        return self._get_matrix_plot(alpha.cpu().unsqueeze(0))

    def _get_matrix_plot(self, matrix):
        if matrix.device != "cpu":
            matrix = matrix.cpu()
        x = numpy.arange(-0.5, matrix.shape[1], 1)
        y = numpy.arange(-0.5, matrix.shape[0], 1)
        fig, ax = plt.subplots()
        # c = ax.pcolormesh(x, y, matrix, cmap='RdBu', norm=colors.Normalize())
        c = ax.pcolormesh(x, y, matrix, cmap="RdBu", vmin=-1, vmax=1)
        fig.colorbar(c, ax=ax)
        return fig

    # @visualize_flag
    # def plot_weight_matrix(self, matrix, tag="default_tag"):
    #    self.writer.add_figure(figure=self._get_matrix_plot(matrix), tag=tag, global_step=TODO

    def alpha_density(self, alphas):
        for i, graph in enumerate(alphas.T):
            fig, ax = plt.subplots()
            n, bins, patches = ax.hist(graph, 15, density=True)
            plt.xlim(-3, 3)
            self.writer.add_figure(
                figure=fig, tag=f"Density for graph {i + 1}", global_step=i + 1
            )
            # n, bins, patches = ax.hist(graph, 50, density=True, facecolor='g', alpha=0.75)
            # plt.xlabel('Smarts')
            # plt.ylabel('Probability')
            # plt.title('Histogram of IQ')
            # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
            # plt.xlim(40, 160)
            # plt.ylim(0, 0.03)
            # plt.grid(True)
            # plt.show()


if __name__ == "__main__":
    settings = {
        "visualize": True,
        "visualizationPath": "../runs/Sep021254_alpha_density",
        "resultPath": "",
    }
    alphas = torch.tensor(
        [
            [-0.6724, -0.6169, -0.1174, 3.9948, -0.4750],
            [0.0530, 0.7437, 1.4537, 2.6420, -2.1995],
            [-0.6479, -0.1244, 0.0658, 3.4020, 1.5757],
            [1.6682, -0.0979, 0.1354, 0.6284, -0.0188],
            [0.9905, 1.1689, 1.1591, -0.4372, 0.3652],
            [-0.3553, -0.3166, -0.5735, 3.4327, 0.5817],
            [-0.4869, -0.8976, -1.8490, 0.6425, 3.4236],
            [0.9223, 0.0608, -0.4997, 0.5071, 0.4761],
            [-0.2119, -0.5833, -1.0623, 0.2192, 3.8338],
            [-0.7795, 0.6106, -0.7030, 1.3432, 0.9752],
            [1.1417, 0.7042, 0.7643, -0.0168, 0.8719],
            [0.5270, 0.4868, 1.2751, 0.9563, 0.5904],
            [0.7536, -0.3478, -0.0372, 1.1353, 0.6663],
            [-0.4143, -0.1952, -0.5577, -0.3836, 3.1966],
            [-0.3807, -0.1790, -0.6172, 3.1132, -0.8820],
            [0.5753, 0.9700, 0.7932, 0.7387, 0.8480],
            [0.9353, 0.5650, -0.4954, 0.1802, 0.6924],
            [0.5211, 0.2047, -1.0372, 1.9160, 1.3444],
            [-0.4065, -0.7774, -0.5614, 3.5175, 0.3807],
            [-0.0686, -0.1831, 0.4119, 1.4453, 2.0071],
            [-0.2247, -0.1118, 0.5120, 0.4369, 1.0707],
            [-0.3241, -0.0784, -0.6062, 1.8726, -0.2023],
            [0.4293, 0.3984, 0.3778, 0.5192, 0.3317],
            [0.4758, 0.0449, 1.1141, -1.1095, 2.1772],
            [0.7009, 0.6671, 0.5160, 0.7170, 0.6232],
            [0.4834, 0.1786, -0.0169, 2.0696, 2.6668],
            [0.8610, 0.1491, 0.1426, 1.4993, 0.7401],
            [-0.2268, 0.3861, -0.1033, 2.0143, 0.3805],
            [0.4470, 0.6380, 0.1765, 0.7228, -0.2764],
            [-0.1412, -0.1391, 1.3774, 2.1892, 0.4261],
        ]
    )
    v = Visualization(settings, [], [])
    v.alpha_density(alphas)
