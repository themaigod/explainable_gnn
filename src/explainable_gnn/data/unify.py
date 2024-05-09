import explainable_gnn as eg


class HINData(eg.Model):
    """
    HINData Class
    ==============

    The ``HINData`` class is designed to manage and process data associated with heterogeneous information networks (HINs). This class supports various data types including graphs, node features, and more, with optional data processing through designated processors.

    Attributes
    ----------
    keys : list
        A list of attribute names that the class can manage: 'graph', 'node_types', 'node_features', 'labels', 'subgraph', 'meta_path'.

    Constructor
    -----------
    .. method:: HINData.__init__(**kargs)

       Initializes a new instance of ``HINData`` with optional data processing. The class accepts various types of graph-related data, each potentially paired with a specific processor for data transformations.

       :param kargs: Keyword arguments dynamically setting the attributes based on the predefined keys. Each key can be associated with a processor and processor arguments to allow for dynamic data transformations at initialization.
           - graph: The main graph data.
           - node_types: Types of the nodes in the graph.
           - node_features: Features associated with the nodes.
           - labels: Labels associated with the nodes.
           - subgraph: Subgraphs derived from the main graph.
           - meta_path: Meta paths used within the graph for specific operations or analyses.
           - *_processor: A callable that processes the data for a corresponding key.
           - *_processor_kargs: Arguments to pass to the processor callable.
           - unified (bool): If True, applies the specified processors to the data upon initialization.

    Methods
    -------
    .. method:: HINData.unified_data()

       Processes all data attributes that have associated processors. If processor arguments are provided, they are passed to the processor function.

    .. method:: HINData.set_processor(processor_target, processor)

       Sets a data processor for a specific attribute.

       :param processor_target: The target attribute for processing.
       :param processor: A callable that will process the data of the target attribute.

    .. method:: HINData.return_data()

       Returns a dictionary of attributes that have been set and are not None.

       :return: A dict of key-value pairs where keys are attribute names and values are the corresponding data.

    .. method:: HINData.return_all()

       Returns a dictionary of all attributes, including processors and their arguments.

       :return: A dict including all keys, processors, and processor arguments.

    .. method:: HINData.check_data()

       A placeholder method for data validation, which can be overridden in subclasses to implement specific data checks.

    Examples
    --------
    Creating an instance of ``HINData`` with graph data and an example processor for node features:

    .. code-block:: python

        import numpy as np
        import scipy.sparse as sp

        graph = sp.csr_matrix([[0, 1], [1, 0]])
        node_features = np.array([[1, 2], [3, 4]])
        labels = np.array([0, 1])

        def scale_features(features):
            return features / np.max(features)

        hin_data = HINData(
            graph=graph,
            node_features=node_features,
            labels=labels,
            node_features_processor=scale_features,
            unified=True
        )

    Notes
    -----
    - The ``HINData`` class is highly customizable and flexible, designed to support various types of data commonly used in heterogeneous information network analysis.
    - Processors allow for modular and reusable data transformations, enhancing the flexibility of data manipulation within the class.

    """
    keys = [
        'graph',
        'node_types',
        'node_features',
        'labels',
        'subgraph',
        'meta_path',
    ]

    def __init__(self, **kargs):
        for k in self.keys:
            if k in kargs:
                setattr(self, k, kargs[k])

            if k + '_processor' in kargs:
                setattr(self, k + '_processor', kargs[k + '_processor'])

            if k + '_processor_kargs' in kargs:
                setattr(self, k + '_processor_kargs', kargs[k + '_processor_kargs'])

        if kargs.get('unified', False):
            self.unified_data()

    def unified_data(self):
        for k in self.keys:
            if getattr(self, k, None) is not None and getattr(self, k + '_processor',
                                                              None) is not None:
                if getattr(self, k + '_processor_kargs', None) is not None:
                    setattr(self, k, getattr(self, k + '_processor')(getattr(self, k),
                                                                     **getattr(self,
                                                                               k + '_processor_kargs')))
                else:
                    setattr(self, k, getattr(self, k + '_processor')(getattr(self, k)))

    def set_processor(self, processor_target, processor):
        processor_name = processor_target + '_processor'
        setattr(self, processor_name, processor)

    def return_data(self):
        return {
            k: getattr(self, k) for k in self.keys if
            getattr(self, k, None) is not None
        }

    def return_all(self):
        data = {
            k: getattr(self, k) for k in self.keys
        }
        for k in self.keys:
            if k + '_processor' in data:
                data[k + '_processor'] = getattr(self, k + '_processor')
            if k + '_processor_kargs' in data:
                data[k + '_processor_kargs'] = getattr(self, k + '_processor_kargs')
        return data

    def check_data(self):
        pass


class DirectedGraphData(eg.Model):
    """
    DirectedGraphData Class
    =======================

    The ``DirectedGraphData`` class handles the storage and processing of data associated with directed graphs. It is designed to be flexible, allowing for dynamic data assignment and processing through user-defined functions.

    Attributes
    ----------
    keys : list
        A list of attribute names managed by the class, specifically: 'graph', 'node_features', and 'labels'.

    Constructor
    -----------
    .. method:: DirectedGraphData.__init__(**kargs)

       Initializes a new instance of ``DirectedGraphData``, setting attributes based on provided keyword arguments and optionally applying data processing functions.

       :param kargs: Keyword arguments for setting attributes and processors:
           - Each key in 'keys' (graph, node_features, labels) can be set by passing it as a keyword argument.
           - Each key may also have an associated '_processor' which is a function intended to modify the data upon setting.
           - '_processor_kargs' are optional keyword arguments for the processors.
           - 'unified' (bool): If True, applies the specified processors to the data immediately upon initialization.

    Methods
    -------
    .. method:: DirectedGraphData.unified_data()

       Applies processors to all data attributes that have an associated processor defined. If processor arguments are provided, they are used when calling the processor.

    .. method:: DirectedGraphData.set_processor(processor_target, processor)

       Sets a processor for a specified attribute.

       :param processor_target: The attribute name (e.g., 'graph', 'node_features', 'labels') to which the processor should be applied.
       :param processor: A callable that processes the specified data.

    .. method:: DirectedGraphData.return_data()

       Returns a dictionary of all attributes that are currently set and not None.

       :return: A dictionary with keys as attribute names and values as the data stored under those attributes.

    .. method:: DirectedGraphData.return_all()

       Returns a dictionary of all attributes including their processors and processor arguments if available.

       :return: A comprehensive dictionary including all data, processors, and processor arguments.

    .. method:: DirectedGraphData.check_data()

       Placeholder method for data validation, which can be implemented in subclasses to enforce specific data integrity rules or checks.

    Examples
    --------
    Creating an instance of ``DirectedGraphData`` with a graph and custom processing for node features:

    .. code-block:: python

        import numpy as np
        import scipy.sparse as sp

        graph = sp.csr_matrix([[0, 1], [1, 0]])
        node_features = np.array([[1, 2], [3, 4]])
        labels = np.array([0, 1])

        def normalize(features):
            return features / np.linalg.norm(features, axis=1, keepdims=True)

        graph_data = DirectedGraphData(
            graph=graph,
            node_features=node_features,
            labels=labels,
            node_features_processor=normalize,
            unified=True
        )

    Notes
    -----
    - This class is particularly useful in applications involving machine learning on graphs where preprocessing of graph data is a common requirement.
    - The flexible architecture of the class allows for easy extension and customization to fit specific needs of various graph processing tasks.

    """
    keys = [
        'graph',
        'node_features',
        'labels',
    ]

    def __init__(self, **kargs):
        for k in self.keys:
            if k in kargs:
                setattr(self, k, kargs[k])

            if k + '_processor' in kargs:
                setattr(self, k + '_processor', kargs[k + '_processor'])

            if k + '_processor_kargs' in kargs:
                setattr(self, k + '_processor_kargs', kargs[k + '_processor_kargs'])

        if kargs.get('unified', False):
            self.unified_data()

    def unified_data(self):
        for k in self.keys:
            if getattr(self, k, None) is not None and getattr(self, k + '_processor',
                                                              None) is not None:
                if getattr(self, k + '_processor_kargs', None) is not None:
                    setattr(self, k, getattr(self, k + '_processor')(getattr(self, k),
                                                                     **getattr(self,
                                                                               k + '_processor_kargs')))
                else:
                    setattr(self, k, getattr(self, k + '_processor')(getattr(self, k)))

    def set_processor(self, processor_target, processor):
        processor_name = processor_target + '_processor'
        setattr(self, processor_name, processor)

    def return_data(self):
        return {
            k: getattr(self, k) for k in self.keys if
            getattr(self, k, None) is not None
        }

    def return_all(self):
        data = {
            k: getattr(self, k) for k in self.keys
        }
        for k in self.keys:
            if k + '_processor' in data:
                data[k + '_processor'] = getattr(self, k + '_processor')
            if k + '_processor_kargs' in data:
                data[k + '_processor_kargs'] = getattr(self, k + '_processor_kargs')
        return data

    def check_data(self):
        pass
