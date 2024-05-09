import numpy as np
import scipy.sparse as sp
import explainable_gnn as eg

from .unify import HINData, DirectedGraphData, Data


class StandardData(Data):
    pass


class StandardHINData(HINData, StandardData):
    """
    StandardHINData Class
    =====================

    The ``StandardHINData`` class, which inherits from ``HINData``, enforces a stricter data format and dimensionality for heterogeneous information networks. It provides additional validation to ensure data conformity and introduces automatic data processing if unified mode is enabled.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        A (num_nodes, num_nodes) adjacency matrix representing the graph structure.
    node_types : numpy.ndarray
        A (num_nodes,) array indicating the type of each node.
    node_features : numpy.ndarray
        A (num_nodes, num_features) array of node features.
    labels : numpy.ndarray
        A (num_nodes, num_classes) array of node labels.
    subgraph : list of scipy.sparse.csr_matrix
        A list of (num_nodes, num_nodes) adjacency matrices, each representing a subgraph.
    meta_path : list of tuples or list of int
        Specifies the meta paths in terms of node types tuples or lists of edge types.
    unified : bool, optional
        If True, all data will be processed upon retrieval to ensure it conforms to the standard formats. Defaults to False.
    *_processor : callable object, optional
        A processing function to be applied to the data.
    *_processor_kargs : dict, optional
        Keyword arguments for the processor function.

    Attributes
    ----------
    standard_type : type
        The standard data type for arrays (numpy.ndarray).
    standard_type_name : str
        The name of the standard data type ("numpy.ndarray").
    standard_type_unify : callable
        A function to convert data to the standard numpy.ndarray format.
    standard_sparse_type : type
        The standard data type for sparse matrices (scipy.sparse.csr_matrix).
    standard_sparse_type_name : str
        The name of the standard sparse data type ("scipy.sparse.csr_matrix").
    standard_sparse_type_unify : callable
        A function to convert data to the standard scipy.sparse.csr_matrix format.
    standard_keys : list of str
        Keys that are expected to conform to ``standard_type``.
    standard_sparse_keys : list of str
        Keys that are expected to conform to ``standard_sparse_type``.
    special_check : str
        A string indicating the type of special check to perform ("dimension").

    Methods
    -------
    check_data()
        Validates the data types of the standard and sparse keys against their expected types.
    special_check_data()
        Checks for special cases as specified by the ``special_check`` attribute.
    check_dimension()
        Verifies that all data dimensions conform to expected standards.
    ask_data(key)
        Retrieves the data associated with the given key if available; otherwise, raises an error.

    Examples
    --------
    Creating a ``StandardHINData`` instance:

    .. code-block:: python

        import explainable_gnn as eg
        data = eg.data.StandardHINData(
            graph=graph,
            node_types=node_types,
            node_features=node_features,
            labels=labels,
            subgraph=subgraph,
            meta_path=meta_path,
            unified=True,
            graph_processor=eg.data.GraphProcessor,
        )

    Notes
    -----
    This class is part of the explainable GNN framework and assumes that all input data is provided in a form that can directly be processed by the framework's functions.
    """
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
        """
        Initializes the StandardHINData object with the provided parameters. All data is
        validated and optionally unified upon initialization.
        """
        super().__init__(**kargs)
        self.check_data()

    def check_data(self):
        """
        Validates the data types of the items stored under standard_keys and standard_sparse_keys.
        For each key, it checks if the data conforms to the expected data type, issuing a warning
        and converting the data if necessary.

        * Raises:
            - None: This method does not raise exceptions but will print warnings to standard output.
        """
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
        Performs additional checks specified by the special_check attribute. Currently, only
        'dimension' checks are implemented, which verify the correct dimensions of the data arrays.

        * Raises:
            - NotImplementedError: If the ``special_check`` attribute specifies a check that is not implemented.
        """
        if self.special_check is not None:
            if self.special_check == "dimension":
                self.check_dimension()
            else:
                raise NotImplementedError(
                    f"Special check {self.special_check} is not implemented")

    def check_dimension(self):
        """
        Ensures that all data dimensions are as expected. This method checks the dimensions of ``node_features``, ``labels``, ``graph``, ``subgraph``, and ``node_types`` to ensure they meet the dimensional criteria required by the framework.

        * Details:
          - ``node_features`` should be 2-dimensional.
          - ``labels`` should be 2-dimensional.
          - ``graph`` should be 2-dimensional and square.
          - Each ``subgraph`` should be square and have dimensions matching ``graph``.
          - ``node_types`` should be 1-dimensional or a 2-dimensional column vector reshaped to 1-dimension.
          - Ensures that all nodes counts across different data types are consistent.

        * Raises:
          - AssertionError: If any of the dimensional checks fail.
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
        """
        Retrieves the data associated with the given key from the instance, if it exists.

        * Parameters:
          - key (str): The key corresponding to the data attribute to retrieve.

        * Returns:
          - The data associated with the specified key.

        * Raises:
          - ValueError: If the specified key does not exist in the data object or is None.
        """
        if getattr(self, key, None) is not None:
            return getattr(self, key)
        else:
            raise ValueError(f"{key} is required but not found in provided data object")


class MetaPath(eg.Model):
    """
    MetaPath Class
    ==============

    The ``MetaPath`` class encapsulates metadata paths, either defined by node types or edge types within a graph, and optionally integrates experimental features like metagraph structures.

    Constructor
    -----------
    .. method:: MetaPath.__init__(meta_path, meta_path_type="node", **kargs)

       Initializes a new instance of the ``MetaPath`` class.

       :param meta_path: The specification of the meta path.
           - If ``meta_path_type`` is "node", it should be a list of tuples specifying node types.
           - If ``meta_path_type`` is "edge", it should be a list of integers representing edge types.
       :param meta_path_type: (str) Indicates whether the ``meta_path`` consists of node types ("node") or edge types ("edge"). Default is "node".
       :param experimental_feature: (bool) Flag to enable experimental features. Default is False.
       :param metagraph: (bool) Enables the use of a metagraph if experimental features are enabled. Default is False.

       Examples:

       .. code-block:: python

           # Using node type meta paths
           MetaPath([(1, 2), (2, 3)], "node")
           # Using edge type meta paths
           MetaPath([1, 2, 3], "edge")
           # Using experimental metagraph features
           import explainable_gnn as eg
           meta_graph = eg.tree()
           meta_graph.add_node(1, parent=None)
           meta_graph.add_node(2, parent=1)
           meta_graph.add_node(3, parent=1)
           MetaPath(meta_graph, "node", metagraph=True, experimental_feature=True)
           MetaPath(meta_graph, "edge", metagraph=True, experimental_feature=True)

    Attributes
    ----------
    meta_path : list or eg.tree
        Stores the meta path definition, which can be a list of tuples, a list of integers, or a tree structure based on the usage context.
    meta_path_type : str
        Specifies the type of meta path ("node" or "edge").
    experimental_feature : bool
        Indicates whether experimental features are enabled.
    metagraph : bool
        Indicates the use of a metagraph if experimental features are enabled.

    Special Methods
    ---------------
    .. method:: MetaPath.__str__()

       Returns a string representation of the meta path.

    .. method:: MetaPath.__repr__()

       Returns a detailed string representation of the meta path, useful for debugging.

    .. method:: MetaPath.__eq__(other)

       Checks equality between this ``MetaPath`` instance and another object.

       :param other: The object to compare against.
       :return: True if the objects are equal, False otherwise.

    .. method:: MetaPath.__hash__()

       Returns a hash based on the meta path.

    .. method:: MetaPath.__set_name__(owner, name)

       Sets the name of the meta path.

       :param owner: The owner class.
       :param name: The name to set.

    .. method:: MetaPath.__set__(instance, value)

       Sets the meta path in an instance.

       :param instance: The instance where the meta path is set.
       :param value: The new value for the meta path.

    .. method:: MetaPath.__get__(instance, owner)

       Gets the meta path from an instance.

       :param instance: The instance from which to get the meta path.
       :param owner: The owner class.

    Methods
    -------
    .. method:: MetaPath.check_data()

       Validates the format of the meta path based on the specified ``meta_path_type`` and the experimental features.

       * Raises:
         - AssertionError: If the meta path format does not conform to the expected type when experimental features are enabled.

    """
    def __init__(self, meta_path, meta_path_type="node", **kargs):
        self.meta_path = meta_path
        self.meta_path_type = meta_path_type
        self.experimental_feature = kargs.get("experimental_feature", False)
        self.metagraph = kargs.get("metagraph", False)

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
        if self.experimental_feature:
            if self.metagraph:
                assert isinstance(self.meta_path,
                                  eg.tree), 'meta_path should be tree when metagraph is True'
            return
        if self.meta_path_type == "node":
            for mp in self.meta_path:
                assert isinstance(mp,
                                  tuple), 'meta_path should be list of node types tuples'
        elif self.meta_path_type == "edge":
            for mp in self.meta_path:
                assert isinstance(mp,
                                  int), 'meta_path should be list of int (edge type)'


class StandardDirectedGraphData(DirectedGraphData, StandardData):

    """
    StandardDirectedGraphData Class
    ===============================

    The ``StandardDirectedGraphData`` class extends the ``DirectedGraphData`` class to ensure that graph data conforms to standard data types and dimensions required for processing in graph neural networks or similar applications.

    Attributes
    ----------
    standard_type : type
        The standard data type for non-sparse data, set to ``numpy.ndarray``.
    standard_type_name : str
        A human-readable name for the standard data type, "numpy.ndarray".
    standard_type_unify : function
        A function used to convert data to the standard non-sparse data type, ``numpy.array``.
    standard_sparse_type : type
        The standard data type for sparse data, set to ``scipy.sparse.csr_matrix``.
    standard_sparse_type_name : str
        A human-readable name for the standard sparse data type, "scipy.sparse.csr_matrix".
    standard_sparse_type_unify : function
        A function used to convert data to the standard sparse data type, ``scipy.sparse.csr_matrix``.
    standard_keys : list
        A list of keys for attributes that should conform to ``standard_type``. Includes ``'node_features'`` and ``'labels'``.
    standard_sparse_keys : list
        A list of keys for attributes that should conform to ``standard_sparse_type``. Includes ``'graph'``.
    special_check : str
        Specifies a special check to perform on the data. Current implementation supports "dimension" for dimensionality checks.

    Constructor
    -----------
    .. method:: StandardDirectedGraphData.__init__(**kargs)

       Initializes a new instance of ``StandardDirectedGraphData``, ensuring that all provided data conforms to specific standards. This class extends ``DirectedGraphData`` by adding additional type checks and data dimension validation.

       The constructor initializes attributes based on the keys defined in the base class and performs type and dimension checks. Each key can also have a corresponding processor and processor arguments for data pre-processing.

       :param kargs: Keyword arguments for dynamically setting graph data attributes. Expected keys include:
           - graph (scipy.sparse.csr_matrix): A sparse matrix representing the adjacency matrix of the graph.
           - node_features (numpy.ndarray): A 2D array where each row represents features of a node.
           - labels (numpy.ndarray): A 2D array where each row represents the label(s) associated with a node.
           - *_processor (callable, optional): Functions to process each corresponding data attribute.
           - *_processor_kargs (dict, optional): Arguments for each processor function.
           - unified (bool, optional): If True, applies the specified processors to the data upon initialization.

       After initializing data attributes and processors, this constructor also checks whether the data should be unified immediately based on the 'unified' keyword argument. Subsequently, it validates the data types and dimensions using the `check_data` method of this class.

       Example:
           To create an instance of ``StandardDirectedGraphData`` with specific graph data and a processor for node features:

           .. code-block:: python

               import scipy.sparse as sp
               import numpy as np

               graph_data = sp.csr_matrix([[0, 1], [1, 0]])
               features = np.array([[1, 2], [3, 4]])
               labels = np.array([0, 1])

               def normalize_features(features):
                   norm = np.linalg.norm(features, axis=1, keepdims=True)
                   return features / norm

               data_instance = StandardDirectedGraphData(
                   graph=graph_data,
                   node_features=features,
                   labels=labels,
                   node_features_processor=normalize_features,
                   unified=True
               )

    Notes:
       - The `__init__` method in `StandardDirectedGraphData` utilizes the flexible structure of the base class to allow for easy customization and extension of data processing routines.
       - Detailed data validation ensures that all data components not only exist but conform to expected dimensions and types, making this class useful in environments where strict data integrity is necessary.


    Methods
    -------
    .. method:: StandardDirectedGraphData.check_data()

       Checks all data attributes against their expected data types and converts them if necessary.

    .. method:: StandardDirectedGraphData.special_check_data()

       Performs special checks on the data as defined by ``special_check``. Currently, this method supports dimensionality checks.

    .. method:: StandardDirectedGraphData.check_dimension()

       Ensures that the dimensions of the data attributes are as expected:
       - ``node_features`` should be 2-dimensional (num_nodes, num_features).
       - ``labels`` should be 2-dimensional (num_nodes, num_classes).
       - ``graph`` should be 2-dimensional (num_nodes, num_nodes).

    Examples
    --------
    Creating an instance of ``StandardDirectedGraphData`` and validating its structure:

    .. code-block:: python

        import numpy as np
        import scipy.sparse as sp
        node_features = np.random.rand(10, 5)
        labels = np.random.randint(0, 2, size=(10, 2))
        graph = sp.rand(10, 10, density=0.1, format='csr')

        data = StandardDirectedGraphData(
            node_features=node_features,
            labels=labels,
            graph=graph
        )

    Notes
    -----
    - When adding new data types or data structures, additional checks or conversions should be implemented in the ``check_data`` and ``special_check_data`` methods.
    - This class automatically handles the conversion of data to ensure compatibility with processing routines that expect data in standard formats.

    """
    standard_type = np.ndarray
    standard_type_name = 'numpy.ndarray'
    standard_type_unify = np.array

    standard_sparse_type = sp.csr_matrix
    standard_sparse_type_name = 'scipy.sparse.csr_matrix'
    standard_sparse_type_unify = sp.csr_matrix

    standard_keys = [
        'node_features',
        'labels',
    ]

    standard_sparse_keys = [
        'graph',
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

        self.special_check_data()

    def special_check_data(self):
        if self.special_check is not None:
            if self.special_check == "dimension":
                self.check_dimension()
            else:
                raise NotImplementedError(
                    f"Special check {self.special_check} is not implemented")

    def check_dimension(self):
        if getattr(self, 'node_features', None) is not None:
            assert len(
                self.node_features.shape) == 2, 'node_features should be 2-dimension'
        if getattr(self, 'labels', None) is not None:
            assert len(self.labels.shape) == 2, 'labels should be 2-dimension'
        if getattr(self, 'graph', None) is not None:
            assert len(self.graph.shape) == 2, 'graph should be 2-dimension'
