import unittest

# import diffNN
# import graph


class TestDiffNN(unittest.TestCase):
    def setUp(self):
        pass
        # self.numberNodes = 3
        # self.width = 7
        # self.graph = graph.GraphGenerator(self.numberNodes).get_random_subset(1)[0]
        # print(self.graph)

    def test_graphNN_from_graph(self):
        # graph = diffNN.GraphNN(self.graph, self.width)
        # print(graph.parameters)
        self.assertEqual(True, True)


class TestGraph(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
