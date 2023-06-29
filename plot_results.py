import functools
import os
import sys

from matplotlib import pyplot as plt
from util import *


class Output:
    def __init__(self, path_parameters, **kwargs):
        self._path_parameters = path_parameters
        self._collectResults(**kwargs)

    def _collectResults(self, **kwargs):
        results = []
        for name_dir in os.listdir(self._path_parameters):
            if not os.path.isdir(os.path.join(self._path_parameters, name_dir)):
                # Ignore non sub-directories (files)
                continue

            for name_file in os.listdir(os.path.join(self._path_parameters, name_dir)):
                path_file = os.path.join(self._path_parameters, name_dir, name_file)
                assert os.path.isfile(path_file)
                with open(path_file, "r+") as handle:
                    settings = Settings(**json.load(handle))
                if settings.check(**kwargs):
                    results.append(
                        persistence.read(
                            [name_file, Constants.name_dir_results, "results.pt"],
                            torch_load=True,
                        )
                    )
        print([result["performance"]["accuracy"][-1] for result in results])
        results = [
            list(zip(result["graphs"][stage], result["alphas"][-1][stage]))
            for result in results
            for stage in result["graphs"]
        ]
        results = list(functools.reduce(lambda a, b: a + b, results))
        self.results = {key: [] for key in [result[0] for result in results]}
        for k, v in results:
            self.results[k].append(v)

    def alphaBoxPlot(self):
        plt.boxplot(list(self.results.values()))
        plt.xticks(range(1, len(self.results.keys()) + 1), list(self.results.keys()))
        plt.show()

    # def plot_graphs(self, graphs):
    #    for i, g in enumerate(graphs):
    #        fig = self._visualize_graph(g)
    #        self.writer.add_figure(figure=fig, tag="_".join(["Graphs"]), global_step=i)

    # def _visualize_graph(self, networkx_graph):
    #    draw_networkx(networkx_graph)
    #    return plt.gcf()


if __name__ == "__main__":
    path_parameters = os.path.join(Constants.path_parameters, sys.argv[1])
    output = Output(path_parameters=path_parameters)
    output.alphaBoxPlot()
