import numpy as np
import scipy.sparse as sp

from .unify import HINData


class StandardData(HINData):
    standard_type = np.ndarray
    standard_type_name = 'numpy.ndarray'
    standard_type_unify = np.array

    standard_sparse_type = sp.csr_matrix
    standard_sparse_type_name = 'scipy.sparse.csr_matrix'
    standard_sparse_type_unify = sp.csr_matrix

    standard_keys = [
        'node_types',
        'node_features',
        'labels',
    ]

    standard_sparse_keys = [
        'graph',
        'subgraph',
    ]

    special_check = "dimension"

    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.check_data()

    def check_data(self):
        for k in self.standard_keys:
            if getattr(self, k, None) is not None and not isinstance(getattr(self, k),
                                                                     self.standard_type):
                print(f'Warning: {k} is not {self.standard_type_name}')
                self.k = self.standard_type_unify(k)

        for k in self.standard_sparse_keys:
            if getattr(self, k, None) is not None and not isinstance(getattr(self, k),
                                                                     self.standard_sparse_type):
                print(f'Warning: {k} is not {self.standard_sparse_type_name}')
                self.k = self.standard_sparse_type_unify(k)

        if getattr(self, 'meta_path', None) is not None:
            assert isinstance(self.meta_path, list) or isinstance(self.meta_path,
                                                                  tuple) or isinstance(
                self.meta_path,
                MetaPath), 'meta_path should be list or tuple or MetaPath'
        self.special_check_data()

    def special_check_data(self):
        """
        Check the data for special cases where point out in special_check
        :return:
        """
        if self.special_check is not None:
            if self.special_check == "dimension":
                self.check_dimension()
            else:
                raise NotImplementedError(
                    f"Special check {self.special_check} is not implemented")

    def check_dimension(self):
        """
        Check the dimension of the data
        Here is the details
        node_features: (num_nodes, num_features)
        labels: (num_nodes, num_classes)
        graph: (num_nodes, num_nodes)
        subgraph: list of (num_nodes, num_nodes)
        node_types: (num_nodes,) or (num_nodes, 1) (reshape to (num_nodes,))
        :return:
        """
        if getattr(self, 'node_features', None) is not None:
            assert len(
                self.node_features.shape) == 2, 'node_features should be 2-dimension'
        if getattr(self, 'labels', None) is not None:
            assert len(self.labels.shape) == 2, 'labels should be 2-dimension'
        if getattr(self, 'graph', None) is not None:
            assert len(self.graph.shape) == 2, 'graph should be 2-dimension'
        if getattr(self, 'subgraph', None) is not None:
            for subgraph in self.subgraph:
                assert len(subgraph.shape) == 2, 'subgraph should be 2-dimension'
        # num_nodes check
        if getattr(self, 'node_features', None) is not None and getattr(self, 'graph',
                                                                        None) is not None:
            assert self.node_features.shape[0] == self.graph.shape[
                0], 'num_nodes should be same'
            assert self.node_features.shape[0] == self.graph.shape[
                1], 'num_nodes should be same'
        if getattr(self, 'node_features', None) is not None and getattr(self, 'labels',
                                                                        None) is not None:
            assert self.node_features.shape[0] == self.labels.shape[
                0], 'num_nodes should be same'
        if getattr(self, 'graph', None) is not None and getattr(self, 'labels',
                                                                None) is not None:
            assert self.graph.shape[0] == self.labels.shape[
                0], 'num_nodes should be same'
        if getattr(self, 'subgraph', None) is not None:
            for subgraph in self.subgraph:
                assert subgraph.shape[0] == subgraph.shape[
                    1], 'subgraph should be square matrix'
            for subgraph in self.subgraph:
                assert self.graph.shape[0] == subgraph.shape[
                    0], 'num_nodes should be same'
        if getattr(self, 'node_types', None) is not None:
            if len(self.node_types.shape) == 2:
                assert self.node_types.shape[1] == 1, 'node_types should be 1-dimension'
                self.node_types = self.node_types.reshape(-1)
            assert len(self.node_types.shape) == 1, 'node_types should be 1-dimension'
            assert self.node_types.shape[0] == self.node_features.shape[
                0], 'num_nodes should be same'
        return True

    def ask_data(self, key):
        if getattr(self, key, None) is not None:
            return getattr(self, key)
        else:
            raise ValueError(f"{key} is required but not found in provided data object")


class MetaPath:
    def __init__(self, meta_path, meta_path_type="node"):
        """
        MetaPath class
        :param meta_path:
        If meta_path_type is "node", meta_path should be list of node types tuples
        If meta_path_type is "edge", meta_path should be list of int (edge type)
        :param meta_path_type: "node" or "edge"
        """
        self.meta_path = meta_path
        self.meta_path_type = meta_path_type

    def __str__(self):
        return str(self.meta_path)

    def __repr__(self):
        return str(self.meta_path)

    def __eq__(self, other):
        return self.meta_path == other.meta_path

    def __hash__(self):
        return hash(str(self))

    def __set_name__(self, owner, name):
        self.meta_path = name

    def __set__(self, instance, value):
        self.meta_path = value

    def __get__(self, instance, owner):
        return str(self)

    def check_data(self):
        if self.meta_path_type == "node":
            for mp in self.meta_path:
                assert isinstance(mp,
                                  tuple), 'meta_path should be list of node types tuples'
        elif self.meta_path_type == "edge":
            for mp in self.meta_path:
                assert isinstance(mp,
                                  int), 'meta_path should be list of int (edge type)'
