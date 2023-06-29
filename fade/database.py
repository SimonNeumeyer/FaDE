import functools
import json
import os
import sqlite3 as sql

from .graph import Graph
from .util import Constants, Singleton, uuid


class Database(metaclass=Singleton):
    def __init__(self):
        self.db = sql.connect(
            os.path.join(Constants.path_data_base, Constants.database_name)
        )
        if not self._table_exists(Constants.table_versioning):
            self._create_table_versioning()
        self.versions = dict(
            self.db.execute(f"SELECT * FROM {Constants.table_versioning}").fetchall()
        )
        self._backup()

    def _backup(self):
        with sql.connect(Constants.database_backup_name) as backup_conn:
            self.db.backup(backup_conn)

    def _table_exists(self, tablename):
        return (
            self.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=(?)",
                (tablename,),
            ).fetchone()
            != None
        )

    def _insert(self, tablename):
        if not self._table_exists(tablename):
            self.create(tablename)

    def _toSqlList(self, l):
        return tuple(list(map(lambda e: str(e), l)))

    def get_samples(
        self, tablename, number_samples=0, filter: dict = {}, accept_less=False
    ):
        assert self._table_exists(tablename), f"Table {tablename} does not exist"
        schema = self._get_schema(tablename)
        filter_sql = (
            ""
            if len(filter) == 0
            else "WHERE "
            + " AND ".join(
                [
                    f"{key} = '{filter[key]}'"
                    if not isinstance(filter[key], list)
                    else f"{key} IN {self._toSqlList(filter[key])}"
                    for key in filter
                ]
            )
        )
        filter_sql += f" LIMIT {number_samples}" if number_samples > 0 else ""
        result = self.db.execute(f"SELECT * FROM {tablename} {filter_sql}").fetchall()
        assert (
            len(result) == number_samples or number_samples == 0 or accept_less
        ), f"Only {len(result)} samples available"
        return [dict(zip(schema, row)) for row in result]

    def _get_schema(self, tablename):
        assert self._table_exists(tablename), f"Table {tablename} does not exist"
        return [
            column[1]
            for column in self.db.execute(f"PRAGMA table_info({tablename})").fetchall()
        ]

    def update_schema(self, tablename, columns):  # columns = {columnname: datatype}
        assert self._table_exists(tablename), f"Table {tablename} does not exist"
        self.versions[tablename] = (
            self.versions[tablename] + 1 if tablename in self.versions else 0
        )
        for col in columns.items():
            self.db.execute(f"ALTER TABLE {tablename} ADD COLUMN {col[0]} {col[1]}")
        self.db.commit()

    def update_schema_delete_column(
        self, tablename, columns
    ):  # columns = list of column names
        assert self._table_exists(tablename), f"Table {tablename} does not exist"
        self.versions[tablename] = (
            self.versions[tablename] + 1 if tablename in self.versions else 0
        )
        for col in columns:
            self.db.execute(f"ALTER TABLE {tablename} DROP COLUMN {col}")
        self.db.commit()

    def create(self, tablename):
        assert not self._table_exists(tablename), f"Table {tablename} already exists"
        self.versions[tablename] = (
            self.versions[tablename] + 1 if tablename in self.versions else 0
        )
        if tablename == Constants.table_results:
            self._create_table_results()
        elif tablename == Constants.table_settings:
            self._create_table_settings()
        elif tablename == Constants.table_graphs:
            self._create_table_graphs()
        else:
            raise NotImplementedError()
        self._update_versions()
        self.db.commit()

    def _update_versions(self):
        self._clear_table(Constants.table_versioning)
        self.insert_versioning(list(self.versions.items()))

    def _clear_table(self, tablename):
        self.db.execute(f"DELETE FROM {tablename}")

    def _drop_table(self, tablename):
        self.db.execute(f"DROP TABLE {tablename}")

    def insert_results(self, results):
        self._insert(Constants.table_results)
        self.db.executemany(
            (
                f"INSERT INTO {Constants.table_results} VALUES ({self.versions[Constants.table_results]},"
                """
                            :graph_id,
                            :settings_id,
                            :stage_index,
                            :graph_alpha,
                            :graph_alphas,
                            :epoch_avg_runtime,
                            :epoch_runtimes,
                            :execution_start,
                            :execution_end,
                            :execution_device,
                            :model_accuracy,
                            :model_accuracies,
                            :model_loss,
                            :model_losses)"""
            ),
            results,
        )
        self.db.commit()

    def insert_graphs(self, graphs):
        tablename = Constants.table_graphs
        self._insert(tablename)
        self.db.executemany(
            (
                f"INSERT INTO {tablename} VALUES ("
                """
                            :id,
                            :technical,
                            :number_vertices,
                            :number_operations)"""
            ),
            graphs,
        )
        self.db.commit()

    def insert_settings(self, settings):
        self._insert(Constants.table_settings)
        self.db.execute(
            (
                f"INSERT INTO {Constants.table_settings} VALUES ({self.versions[Constants.table_settings]},"
                f":{Constants.settings_id},"
                f":{Constants.settings_comment},"
                f":{Constants.dataset},"
                f":{Constants.model_operation},"
                f":{Constants.model_activation},"
                f":{Constants.model_normalize},"
                f":{Constants.model_number_stages},"
                f":{Constants.stage_number_graphs},"
                f":{Constants.stage_shared_weights},"
                f":{Constants.graph_number_vertices},"
                f":{Constants.graph_number_operations},"
                f":{Constants.graph_ids},"
                f":{Constants.darts_sampling},"
                f":{Constants.darts_random_init},"
                f":{Constants.darts_learning_rate_start},"
                f":{Constants.darts_learning_rate_end},"
                f":{Constants.darts_optimizer},"
                f":{Constants.darts_optimization_method},"
                f":{Constants.darts_optimization_ratio},"
                f":{Constants.darts_deactivate_darts},"
                f":{Constants.conv_final_number_channels},"
                f":{Constants.conv_pooling},"
                f":{Constants.conv_kernel_size},"
                f":{Constants.mlp_layer_width},"
                f":{Constants.optimization_number_epochs},"
                f":{Constants.optimization_optimizer},"
                f":{Constants.optimization_batch_size},"
                f":{Constants.optimization_learning_rate_start},"
                f":{Constants.optimization_learning_rate_end},"
                f":{Constants.organizational_experiment},"
                f":{Constants.organizational_experiment_group})"
            ),
            settings,
        )
        self.db.commit()

    def insert_versioning(self, versioning):
        self.db.executemany(
            f"INSERT INTO {Constants.table_versioning} VALUES (?,?)", versioning
        )
        self.db.commit()

    def _create_table_results(self):
        self.db.execute(
            (
                f"CREATE TABLE {Constants.table_results} ("
                """
                        version TEXT,
                        graph_id TEXT,
                        settings_id TEXT,
                        stage_index INTEGER,
                        graph_alpha REAL,
                        graph_alphas TEXT,
                        epoch_avg_runtime REAL,
                        epoch_runtimes TEXT,
                        execution_start TEXT,
                        execution_end TEXT,
                        execution_device TEXT,
                        model_accuracy REAL,
                        model_accuracies TEXT,
                        model_loss REAL,
                        model_losses TEXT
                    );
                """
            )
        )

    def _create_table_settings(self):
        self.db.execute(
            (
                f"CREATE TABLE {Constants.table_settings} ("
                f"{Constants.version} TEXT,"
                f"{Constants.settings_id} TEXT,"
                f"{Constants.settings_comment} TEXT,"
                f"{Constants.dataset} TEXT,"
                f"{Constants.model_operation} TEXT,"
                f"{Constants.model_activation} TEXT,"
                f"{Constants.model_normalize} INTEGER,"
                f"{Constants.model_number_stages} INTEGER,"
                f"{Constants.stage_number_graphs} INTEGER,"
                f"{Constants.stage_shared_weights} INTEGER,"
                f"{Constants.graph_number_vertices} INTEGER,"
                f"{Constants.graph_number_operations} INTEGER,"
                f"{Constants.graph_ids} TEXT,"
                f"{Constants.darts_sampling} INTEGER,"
                f"{Constants.darts_random_init} INTEGER,"
                f"{Constants.darts_learning_rate_start} REAL,"
                f"{Constants.darts_learning_rate_end} REAL,"
                f"{Constants.darts_optimizer} TEXT,"
                f"{Constants.darts_optimization_method} TEXT,"
                f"{Constants.darts_optimization_ratio} REAL,"
                f"{Constants.darts_deactivate_darts} INTEGER,"
                f"{Constants.conv_final_number_channels} INTEGER,"
                f"{Constants.conv_pooling} TEXT,"
                f"{Constants.conv_kernel_size} INTEGER,"
                f"{Constants.mlp_layer_width} INTEGER,"
                f"{Constants.optimization_number_epochs} INTEGER,"
                f"{Constants.optimization_optimizer} TEXT,"
                f"{Constants.optimization_batch_size} INTEGER,"
                f"{Constants.optimization_learning_rate_start} REAL,"
                f"{Constants.optimization_learning_rate_end} REAL,"
                f"{Constants.organizational_experiment} TEXT,"
                f"{Constants.organizational_experiment_group} TEXT"
                ");"
            )
        )

    def _create_table_graphs(self):
        self.db.execute(
            f"CREATE TABLE {Constants.table_graphs} ("
            """
                        id TEXT,
                        technical TEXT,
                        number_vertices INTEGER,
                        number_operations INTEGER
                        );
                        """
        )

    def _one_time_init_graphs(self):
        graphs_total = []
        for k in range(3, 7):
            print(f"Number vertices: {k}")
            graphs = [{}]
            for i in range(1, k):
                one_node_smaller_graphs = graphs
                graphs = []
                for one_node_smaller_graph in one_node_smaller_graphs:
                    for j in range(2**i):
                        graph = one_node_smaller_graph.copy()
                        graph[k - i] = [int(c) for c in f"{j:0{k}b}"]
                        graphs.append(graph)
            graphs = [Graph(json.dumps(g)) for g in graphs]
            cleaned = []
            counter = 0
            for g in graphs:
                counter += 1
                if not any([g.is_isomorphic(h) for h in cleaned]):
                    cleaned.append(g)
                if counter % 100 == 0:
                    print(counter / len(graphs))
            graphs_total.extend(cleaned)
        print("Done creating graphs")
        graphs_total = [
            {
                "id": uuid(time=False),
                "technical": g.get_technical(),
                "number_vertices": g.get_number_vertices(),
            }
            for g in graphs_total
        ]
        self.insert_graphs(graphs_total)

    def _create_table_versioning(self):
        self.db.execute(
            (
                f"CREATE TABLE {Constants.table_versioning} ("
                f"{Constants.versioning_column_table} TEXT, "
                f"{Constants.versioning_column_version} INTEGER"
                f")"
            )
        )
        self.db.commit()

    def count(self, tablename):
        for item in self.db.execute(f"SELECT COUNT(*) FROM {tablename}").fetchall():
            print(item)

    def _input(self):
        self.insert_settings(
            {
                Constants.settings_id: uuid(),
                Constants.settings_comment: "",
                Constants.dataset: "MNIST",
                Constants.model_operation: Constants.OPERATION_CONV,
                Constants.model_activation: Constants.ACTIVATION_RELU,
                Constants.model_normalize: 1,
                Constants.model_number_stages: 1,
                Constants.stage_number_graphs: 1,
                Constants.stage_shared_weights: False,
                Constants.graph_number_vertices: 5,
                Constants.graph_number_operations: 7,
                Constants.graph_ids: json.dumps([]),
                Constants.darts_sampling: True,
                Constants.darts_random_init: True,
                Constants.darts_learning_rate_start: 0.05,
                Constants.darts_learning_rate_end: 0.01,
                Constants.darts_optimizer: Constants.OPTIMIZER_ADAM,
                Constants.darts_optimization_method: Constants.darts_optimization_method_trivial,
                Constants.darts_optimization_ratio: 0,
                Constants.darts_deactivate_darts: 1,
                Constants.conv_final_number_channels: 32,
                Constants.conv_pooling: Constants.POOLING_MAX,
                Constants.conv_kernel_size: 3,
                Constants.mlp_layer_width: 17,
                Constants.optimization_number_epochs: 20,
                Constants.optimization_optimizer: Constants.OPTIMIZER_ADAM,
                Constants.optimization_batch_size: 257,
                Constants.optimization_learning_rate_start: 0.05,
                Constants.optimization_learning_rate_end: 0.01,
                Constants.organizational_experiment: "testconv",
                Constants.organizational_experiment_group: "testconv",
            }
        )

    def count_attribute(self, tablename, column):
        def red(x, y):
            ops = str(y[column])
            if ops in x:
                x[ops] += 1
            else:
                x[ops] = 1
            return x

        data = self.get_samples(tablename)
        return functools.reduce(red, data, {})


if __name__ == "__main__":
    settings = {
        Constants.settings_id: uuid(),
        Constants.settings_comment: "",
        Constants.dataset: "MNIST",
        Constants.model_operation: Constants.OPERATION_LINEAR,
        Constants.model_activation: Constants.ACTIVATION_RELU,
        Constants.model_normalize: 1,
        Constants.model_number_stages: 1,
        Constants.stage_number_graphs: 1,
        Constants.stage_shared_weights: False,
        Constants.graph_number_vertices: 5,
        Constants.graph_number_operations: 7,
        Constants.graph_ids: json.dumps(["c26e8dda57724581898c7a3845f28582"]),
        Constants.darts_sampling: True,
        Constants.darts_random_init: True,
        Constants.darts_learning_rate_start: 0,
        Constants.darts_learning_rate_end: 0.01,
        Constants.darts_optimizer: Constants.OPTIMIZER_ADAM,
        Constants.darts_optimization_method: Constants.darts_optimization_method_trivial,
        Constants.darts_optimization_ratio: 0,
        Constants.darts_deactivate_darts: 1,
        Constants.conv_final_number_channels: 32,
        Constants.conv_pooling: Constants.POOLING_MAX,
        Constants.conv_kernel_size: 3,
        Constants.mlp_layer_width: 0,
        Constants.optimization_number_epochs: 20,
        Constants.optimization_optimizer: Constants.OPTIMIZER_ADAM,
        Constants.optimization_batch_size: 257,
        Constants.optimization_learning_rate_start: 0,
        Constants.optimization_learning_rate_end: 0.01,
        Constants.organizational_experiment: "gridsearch",
        Constants.organizational_experiment_group: "",
    }
    for lr in [0.05, 1]:
        for w in [17, 43, 73]:
            for n in [0, 1]:
                settings.update(
                    {
                        Constants.settings_id: uuid(),
                        Constants.darts_learning_rate_start: lr,
                        Constants.optimization_learning_rate_start: lr,
                        Constants.mlp_layer_width: w,
                        Constants.model_normalize: n,
                        Constants.organizational_experiment_group: "gridsearch"
                        + str(n),
                    }
                )
                Database().insert_settings(settings)
