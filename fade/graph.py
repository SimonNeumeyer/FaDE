import json

import matplotlib.pyplot as plt
import numpy
from networkx import DiGraph, draw_networkx
from networkx import is_isomorphic as lib_is_isomorphic


class Graph:
    def __init__(self, id, technical):
        self.id = id
        self.technical = technical
        vertices, edges = self._contain_from_technical(technical)
        self.lib_graph = self._create_DiGraph(vertices, edges)
        self.vertices_except_input_output = list(self.lib_graph.nodes)
        self.vertices_except_input_output.sort()
        self.input_node = "INPUT"
        self.output_node = "OUTPUT"
        self._append_input_output_node()

    def _contain_from_technical(self, technical):
        technical = json.loads(technical)
        edges = []
        vertices = numpy.array(range(1, len(technical) + 2))
        for v in technical:
            edges.extend(
                [(v, str(s)) for s in vertices[numpy.array(technical[v], dtype=bool)]]
            )
        return ([str(v) for v in vertices], edges)

    def _create_DiGraph(self, vertices, edges):
        graph = DiGraph()
        graph.add_nodes_from(vertices)
        graph.add_edges_from(edges)
        return graph

    def _append_input_output_node(self):
        self.lib_graph.add_edges_from(
            [
                (self.input_node, v)
                for v in self.lib_graph.nodes()
                if not any(True for _ in self.lib_graph.predecessors(v))
            ]
        )
        self.lib_graph.add_edges_from(
            [
                (v, self.output_node)
                for v in self.lib_graph.nodes()
                if not any(True for _ in self.lib_graph.successors(v))
            ]
        )

    def is_isomorphic(self, other):
        assert isinstance(other, Graph), "Parameter has to be of type graph"
        return lib_is_isomorphic(self.get_networkx(), other.get_networkx())

    # hacky
    def get_dense_edges(self):
        graph = Graph(id="", technical=self.technical)
        vertices = self.get_ordered_nodes(except_input_output=False)
        edges = []
        for i, v in enumerate(vertices[:-1]):
            edges.extend([(v, s) for s in vertices[i + 1 :]])
        graph.lib_graph = self._create_DiGraph(vertices, edges)
        graph.vertices_except_input_output = self.get_ordered_nodes()
        return graph.get_edges()

    def get_id(self):
        return self.id

    def get_technical(self):
        return self.technical

    def get_number_vertices(self):
        return len(self.lib_graph.nodes) - 2

    def get_number_operations(self):
        return len([edge for edge in self.get_edges() if edge[0] != self.input_node])

    def visualize(self, ax=None):
        draw_networkx(self.get_networkx(), ax=ax)

    def get_predecessors(self, v_to):
        assert v_to in self.lib_graph.nodes(), "Node not contained in graph"
        return self.lib_graph.predecessors(v_to)

    def is_input_edge(self, v_from, v_to):
        return v_from == self.input_node

    def get_networkx(self):
        return self.lib_graph

    def get_edges(self):
        return self.lib_graph.edges

    def get_ordered_nodes(self, except_input_node=True):
        if not except_input_node:
            return (
                [self.input_node]
                + self.vertices_except_input_output
                + [self.output_node]
            )
        else:
            return self.vertices_except_input_output + [self.output_node]

    # def set_edge_attribute(self, v_from, v_to, key, value):
    #    assert all([v in self.lib_graph.nodes for v in [v_from, v_to]]), "Edge to manipulate is not contained in graph"
    #    self.lib_graph[v_from][v_to][key] = value

    # def get_edge_attribute(self, v_from, v_to):
    #    assert all([v in self.lib_graph.nodes for v in [v_from, v_to]]), "Edge to manipulate is not contained in graph"
    #    return self.lib_graph[v_from][v_to]
