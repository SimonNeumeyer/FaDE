import json
import sys

from fade.data import MNIST
from fade.database import Database
from fade.graph import Graph
from fade.model import *
from fade.train import Trainer
from fade.util import Results, timestamp
from fade.visualization import Visualization


# first bash parameter specifies name of experiment setting
def args_experiment():
    return sys.argv[1] if len(sys.argv) > 1 else None


# second bash parameter specifies number of repititions
def args_repititions():
    return int(sys.argv[2]) if len(sys.argv) > 2 else 1


# rest bash parameter specify graph ids
def args_graphs():
    return sys.argv[3:] if len(sys.argv) > 3 else []


def test(x):
    return x


python_filter = filter

if __name__ == "__main__":
    # Settings
    all_settings = Database().get_samples(
        Constants.table_settings,
        number_samples=1,
        filter={Constants.organizational_experiment_group: args_experiment()}
        if args_experiment()
        else {},
    )
    for settings in all_settings:
        settings[Constants.graph_ids] = (
            json.dumps(args_graphs())
            if args_graphs()
            else settings[Constants.graph_ids]
        )
        settings[Constants.graph_ids] = (
            json.loads(settings[Constants.graph_ids])
            if settings[Constants.graph_ids]
            else []
        )
        for l in range(args_repititions()):
            print(f"\nExperiment repitition {l+1}")
            # Graphs:
            filter = {}
            if settings[Constants.graph_number_vertices] != 0:
                filter[Constants.graph_column_number_vertices] = settings[
                    Constants.graph_number_vertices
                ]
            if settings[Constants.graph_number_operations] != 0:
                filter[Constants.graph_column_number_operations] = settings[
                    Constants.graph_number_operations
                ]
            if settings[Constants.graph_ids]:
                filter[Constants.graph_column_id] = (
                    settings[Constants.graph_ids]
                    if len(settings[Constants.graph_ids]) > 1
                    else settings[Constants.graph_ids][0]
                )
            graphs = Database().get_samples(
                Constants.table_graphs,
                number_samples=settings[Constants.stage_number_graphs],
                filter=filter,
                accept_less=True,
            )
            if (
                len(graphs) < settings[Constants.stage_number_graphs]
                and settings[Constants.graph_ids]
            ):
                filter.pop(Constants.graph_column_id)
                more_graphs = Database().get_samples(
                    Constants.table_graphs,
                    number_samples=settings[Constants.stage_number_graphs],
                    filter=filter,
                    accept_less=True,
                )
                more_graphs = list(
                    python_filter(
                        lambda g: g[Constants.graph_column_id]
                        not in map(lambda e: e[Constants.graph_column_id], graphs),
                        more_graphs,
                    )
                )[: settings[Constants.stage_number_graphs] - len(graphs)]
                graphs += more_graphs
            assert (
                len(graphs) == settings[Constants.stage_number_graphs]
            ), "Not enough graphs fulfil filter criteria"
            graphs = [
                Graph(
                    id=g[Constants.graph_column_id],
                    technical=g[Constants.graph_column_technical],
                )
                for g in graphs
            ]
            # Dataset
            if settings[Constants.dataset] == Constants.dataset_mnist:
                dataset = MNIST()
            else:
                raise NotImplementedError(
                    f"Dataset {settings[Constants.dataset]} not supported"
                )

            # Model
            if settings[Constants.model_operation] == Constants.OPERATION_LINEAR:
                model = MLP(
                    settings=settings,
                    graphs=graphs,
                    dataset_number_channels=dataset.get_number_channels(),
                    dataset_number_features=dataset.get_number_features(),
                    dataset_number_classes=dataset.get_number_classes(),
                )
            elif settings[Constants.model_operation] == Constants.OPERATION_CONV:
                model = ConvolutionalModel(
                    settings=settings,
                    graphs=graphs,
                    dataset_number_channels=dataset.get_number_channels(),
                    dataset_number_features=dataset.get_number_features(),
                    dataset_number_classes=dataset.get_number_classes(),
                )
            else:
                raise NotImplementedError(
                    f"Model {settings[Constants.model_operation]} not supported"
                )
            v = Visualization(flag=True)
            v.plot_model(model, sample=torch.unsqueeze(dataset.get_sample()[0], dim=0))
            v.close()

            # Logger
            results = Results()
            results.log_settings(settings)
            results.log_number_stages(model.get_number_stages())
            for id in model.get_graphNN_ids():
                results.log_graph(
                    graphNN_id=id,
                    graph_id=model.get_graphNN(id).get_graph_id(),
                    stage_index=model.get_stage_index(id),
                )

            # Optimization
            trainer = Trainer(settings, dataset, model, results)
            results.log_execution_start(timestamp())
            trainer.run()
            results.log_execution_end(timestamp())

            # Write results
            Database().insert_results(results.get())
