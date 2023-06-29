import sys
import persistence
import functools
from matplotlib import pyplot as plt
from util import *


class Output:
    def __init__(self, **kwargs):
        self._collectResults(**kwargs)

    def _args(self):
        if len(sys.argv) > 1:
            persistence.dir_parameters = persistence.path([persistence.dir_parameters, sys.argv[1]])

    def _collectResults(self, **kwargs):
        self._args()
        results = []
        for dir in persistence.iterdir(persistence.dir_parameters):
            assert(persistence.is_dir(dir))
            dir_name = persistence.name(dir)
            for file in persistence.iterdir([dir, persistence.folder_settings]):
                assert(not persistence.is_dir(file))
                file_name = persistence.name(file)
                settings = Settings(**json.loads(persistence.read(file)))
                if (settings.check(**kwargs)):
                    results.append(persistence.read([dir, persistence.folder_results, "results.pt"], torch_load=True))
        print([result["performance"]["accuracy"][-1] for result in results])
        results = [list(zip(result['graphs'][stage], result['alphas'][-1][stage])) for result in results for stage in result['graphs']]
        results = list(functools.reduce(lambda a, b: a + b, results))
        self.results = {key: [] for key in [result[0] for result in results]}
        for k, v in results:
            self.results[k].append(v)

    def alphaBoxPlot(self):
        plt.boxplot(list(self.results.values()))
        plt.xticks(range(1, len(self.results.keys()) + 1), list(self.results.keys()))
        plt.show()

    #def plot_graphs(self, graphs):
    #    for i, g in enumerate(graphs):
    #        fig = self._visualize_graph(g)
    #        self.writer.add_figure(figure=fig, tag="_".join(["Graphs"]), global_step=i)

    #def _visualize_graph(self, networkx_graph):
    #    draw_networkx(networkx_graph)
    #    return plt.gcf()


if __name__ == "__main__":
    output = Output()
    output.alphaBoxPlot()
