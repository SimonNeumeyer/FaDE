import datetime
import json
import numbers
import os
from statistics import mean
from timeit import default_timer as timer
from uuid import uuid4

from torch.profiler import ProfilerActivity, profile, record_function


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class InfiniteIterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __iter__(self):
        return iter(self.iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except Exception:
            self.iterator = iter(self.iterable)
            return next(self.iterator)


class Constants(metaclass=Singleton):
    ALPHA = "ALPHA"
    REDUCE_FUNC_SUM = "REDUCE_FUNC_SUM"
    OPTIMIZER_ADAM = "ADAM"
    ACTIVATION_RELU = "ACTIVATION_RELU"
    OPERATION_LINEAR = "OPERATION_LINEAR"
    OPERATION_CONV = "OPERATION_CONV"
    OPERATION_TEST = "OPERATION_TEST"
    POOLING_MAX = "POOLING_MAX"
    POOLING_AVG = "POOLING_AVG"
    FINAL_ALPHAS = "FINAL_ALPHAS"
    dataset_mnist = "MNIST"
    database_name = "database.db"
    database_backup_name = "database_backup.db"
    darts_optimization_method_first_order = "first order"
    darts_optimization_method_second_order = "second order"
    darts_optimization_method_trivial = "trivial"

    # Database
    version = "version"
    table_settings = "SETTINGS"
    table_graphs = "GRAPHS"
    table_results = "RESULTS"
    table_versioning = "VERSIONING"

    # Settings
    settings_id = "settings_id"
    settings_comment = "settings_comment"
    dataset = "dataset"
    model_operation = "model_operation"
    model_activation = "model_activation"
    model_normalize = "model_normalize"
    model_number_stages = "model_number_stages"
    stage_number_graphs = "stage_number_graphs"
    stage_shared_weights = "stage_shared_weights"
    graph_number_vertices = "graph_number_vertices"
    graph_number_operations = "graph_number_operations"
    graph_ids = "graph_ids"
    darts_sampling = "darts_sampling"
    darts_random_init = "darts_random_init"
    darts_learning_rate_start = "darts_learning_rate_start"
    darts_learning_rate_end = "darts_learning_rate_end"
    darts_optimizer = "darts_optimizer"
    darts_optimization_method = "darts_optimization_method"
    darts_optimization_ratio = "darts_optimization_ratio"
    darts_deactivate_darts = "darts_deactivate_darts"
    conv_final_number_channels = "conv_final_number_channels"
    conv_pooling = "conv_pooling"
    conv_kernel_size = "conv_kernel_size"
    mlp_layer_width = "mlp_layer_width"
    optimization_number_epochs = "optimization_number_epochs"
    optimization_optimizer = "optimization_optimizer"
    optimization_batch_size = "optimization_batch_size"
    optimization_learning_rate_start = "optimization_learning_rate_start"
    optimization_learning_rate_end = "optimization_learning_rate_end"
    organizational_worker = "organizational_worker"
    organizational_experiment = "organizational_experiment"
    organizational_experiment_group = "organizational_experiment_group"

    # Graphs
    graph_column_number_vertices = "number_vertices"
    graph_column_number_operations = "number_operations"
    graph_column_id = "id"
    graph_column_technical = "technical"

    # Versioning
    versioning_column_table = "tablename"
    versioning_column_version = "version"

    # Paths
    path_mnist = "/media/data/set/mnist/"
    path_data_base = "data/"
    path_parameters = os.path.join(path_data_base, "parameters")
    path_visualization = os.path.join(path_data_base, "runs")
    name_dir_results = "results"
    name_dir_settings = "settings"


def uuid(time=False):
    if time:
        return datetime.datetime.now().strftime("%b%d%H%M") + str(uuid4()).translate(
            {ord(c): "" for c in "-"}
        )
    else:
        return str(uuid4()).translate({ord(c): "" for c in "-"})


def timestamp():
    return datetime.datetime.now().strftime("%Y%d%H%M")


def copy_tensor(tensor):
    return tensor.data.clone()


def get_numeric(tensor):
    if isinstance(tensor, numbers.Number):
        return tensor
    else:
        return tensor.tolist()


def profile_decorator(func):
    def wrapper_func(self, *args, **kwargs):
        with profile(
            activities=[ProfilerActivity.CPU], record_shapes=False, profile_memory=False
        ) as profiler:
            with record_function(func.__name__):
                result = func(self, *args, **kwargs)
        print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        return result

    return wrapper_func


def speed_decorator(func):
    def wrapper_func(*args, **kwargs):
        start = timer()
        result = func(*args, **kwargs)
        runtime = timer() - start
        if "RUNTIME_LOGGER" in kwargs:
            kwargs["RUNTIME_LOGGER"](runtime)
        # setattr(func, 'runtime', runtime)
        # print(f"Runtime for '{func.__name__}': {runtime}")
        return result

    return wrapper_func


class Results:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.graphs = {}
        self.epoch_runtimes = []
        self.model_accuracies = []
        self.model_losses = []

    def log_lr_weights(self, lr):
        if self.verbose:
            print(f" LR weights: {lr}")

    def log_lr_alphas(self, lr):
        if self.verbose:
            print(f" LR alphas: {lr}")

    def log_epoch(self, epoch):
        if self.verbose:
            print(f"Epoch: {epoch}/{self.number_epochs}")

    def log_settings(self, settings):
        self.settings_id = settings[Constants.settings_id]
        self.number_epochs = settings[Constants.optimization_number_epochs]

    def log_device(self, execution_device):
        self.execution_device = execution_device
        if self.verbose:
            print(f"Device: {execution_device}")

    def log_execution_start(self, start):
        self.execution_start = start

    def log_execution_end(self, end):
        self.execution_end = end

    def log_number_stages(self, number_stages):
        self.model_number_stages = number_stages

    def log_graph(self, graphNN_id, graph_id, stage_index):
        self.graphs[graphNN_id] = {
            "graph_id": graph_id,
            "stage_index": stage_index,
            "alphas": [],
        }

    def log_alpha(self, graphNN_id, alpha):
        self.graphs[graphNN_id]["alphas"].append(alpha)

    def log_runtime(self, runtime):
        self.epoch_runtimes.append(runtime)

    def log_accuracy(self, accuracy):
        self.model_accuracies.append(accuracy)
        if self.verbose:
            print(f" Accuracy: {accuracy}")

    def log_loss(self, loss):
        self.model_losses.append(loss)
        if self.verbose:
            print(f" Loss: {loss}")

    def get(self):
        results = []
        for graph in self.graphs.values():
            results.append(
                {
                    "graph_id": graph["graph_id"],
                    "settings_id": self.settings_id,
                    "stage_index": graph["stage_index"],
                    "graph_alpha": graph["alphas"][-1],
                    "graph_alphas": json.dumps(graph["alphas"]),
                    "epoch_avg_runtime": mean(self.epoch_runtimes),
                    "epoch_runtimes": json.dumps(self.epoch_runtimes),
                    "execution_start": self.execution_start,
                    "execution_end": self.execution_end,
                    "execution_device": self.execution_device,
                    "model_accuracy": self.model_accuracies[-1],
                    "model_accuracies": json.dumps(self.model_accuracies),
                    "model_loss": self.model_losses[-1],
                    "model_losses": json.dumps(self.model_losses),
                }
            )
        return results
