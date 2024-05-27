import torch.nn as nn
from abc import ABC


class BaseModule(ABC):
    """
    BaseModule Class
    ================

    The ``BaseModule`` class serves as an abstract base class (ABC) for all module types within a system that integrates with PyTorch's neural network module system. This class ensures that all modules conform to a required interface and can be recognized and handled uniformly across the framework.

    Description
    -----------
    ``BaseModule`` acts as a foundational component that standardizes module functionality by ensuring that any module, particularly those derived from ``torch.nn.Module``, is registered and acknowledged as a subclass of ``BaseModule``. This registration is crucial for maintaining consistency and interoperability within libraries that depend on these types.

    Purpose
    -------
    The primary purpose of the ``BaseModule`` is to:
    - Provide a common base class for all neural network modules, facilitating the implementation of generic functions and methods that operate on a wide range of modules.
    - Ensure that all modules integrated into the framework are compatible with PyTorch's ``nn.Module``, allowing seamless usage of both custom and PyTorch-provided modules.

    Usage
    -----
    To utilize this class, developers should ensure that any custom module classes they create are registered as subclasses of ``BaseModule`` if they also inherit from ``torch.nn.Module``. This registration can be explicitly stated in the module's class definition or handled programmatically as shown in the class setup.

    Code Example
    ------------
    .. code-block:: python

        import torch.nn as nn

        class MyCustomModule(nn.Module, BaseModule):
            def __init__(self):
                super().__init__()
                # module initialization logic here

    Alternatively, for automatic handling:

    .. code-block:: python

        # Registering torch.nn.Module as a subclass of BaseModule to ensure compatibility
        BaseModule.register(nn.Module)

        class AnotherCustomModule(nn.Module):
            def __init__(self):
                super().__init__()
                # additional setup here

    Notes
    -----
    - While Python's ABC does not enforce the instantiation checks or subclass checks at runtime by default, using the ``register`` method of an ABC allows for the explicit declaration of intent that a certain class should be considered a subclass of the ABC, even if it does not directly inherit from it.
    - This approach is particularly useful in large systems where modules from various sources (internal, external, third-party) need to interact seamlessly, ensuring that all modules meet a baseline level of functionality and compatibility.

    .. note::
       It is critical that all main input modules, especially those directly used within the broader framework or system, are registered as subclasses of ``BaseModule`` to maintain system integrity and module interoperability.

    Conclusion
    ----------
    The ``BaseModule`` class is a strategic implementation detail designed to enhance module management and integration within PyTorch-based systems, promoting a more organized and reliable development environment.

    """
    pass


BaseModule.register(nn.Module)


class Model(BaseModule):
    """
    Model Class
    ===========

    The ``Model`` class is a specialized subclass of the ``BaseModule`` intended to serve as the foundational class for all models in the Explainable GNN framework. This class helps in identifying and managing models, ensuring that all model types conform to a standardized framework structure.

    Purpose
    -------
    The main purpose of the ``Model`` class is to:
    - Provide a recognizable base for all models developed within or for the Explainable GNN framework.
    - Facilitate the easy identification and categorization of models as part of the system, enhancing manageability and compatibility across various components of the framework.
    - Ensure that all models inherit from ``BaseModule``, leveraging the robustness and features established by the base class.

    Usage
    -----
    As a base class, ``Model`` should be extended by any specific model implementations within the Explainable GNN framework. It does not implement specific functionalities but ensures that any subclass aligns with the framework's requirements and standards.

    Developers should subclass ``Model`` when creating new model types to maintain consistency and interoperability within the framework:

    .. code-block:: python

        class MyCustomModel(Model):
            def __init__(self):
                super().__init__()
                # Initialization and model-specific logic here

    By using ``Model`` as the base class, developers ensure that their custom models integrate seamlessly with the framework and other components expecting instances derived from ``Model`` or ``BaseModule``.

    Example
    -------
    Creating a new model that inherits from ``Model``:

    .. code-block:: python

        class ExampleModel(Model):
            def __init__(self):
                super().__init__()
                # Additional setup and layer definitions

    This subclassing makes ``ExampleModel`` a part of the broader Explainable GNN model ecosystem, automatically incorporating it into the framework's module management and processing pipelines.

    Notes
    -----
    - Subclassing ``Model`` is crucial for developers aiming to contribute new model types to the Explainable GNN framework or for those developing proprietary models based on the framework.
    - The ``Model`` class acts as a marker within the framework, indicating that the class is intended for use as a model rather than a general-purpose module.

    Conclusion
    ----------
    The ``Model`` class is strategically important for maintaining a structured and unified approach to model development within the Explainable GNN framework. It ensures that all models adhere to the framework's design principles and are compatible with its architecture and operational protocols.

    """
    pass


class Module(nn.Module, Model):
    """
    Module Class
    ============

    The ``Module`` class is the foundational class for all neural network models in the Explainable GNN framework. It extends both PyTorch's ``nn.Module`` for deep learning functionalities and the custom ``Model`` for compatibility and standardization within the framework.

    Purpose
    -------
    The main purpose of the ``Module`` class is to provide a standardized base that includes not only the functionalities of PyTorch’s computational graphs but also additional methods that are crucial for explainability, saving, loading, and transforming models for inference and deployment.

    Constructor
    -----------
    .. method:: Module.__init__(*args, **kwargs)

       Initializes a new instance of the ``Module`` class, setting up necessary properties and configurations for further extension and use within the framework.

       :param args: Positional arguments passed to the superclass initializers.
       :param kwargs: Keyword arguments passed to the superclass initializers.

    Methods
    -------
    .. method:: Module.parameters_calculation(*args, **kwargs)

       A method that should be implemented to handle the recalculation of parameters when transitioning from one module configuration to another, especially in translation processes.

       :raises NotImplementedError: Indicates that this method needs to be implemented in subclasses.

    .. method:: Module.approximate(*args, **kwargs)

       Should provide the functionality to approximate the model's operations, useful in scenarios requiring model simplification or optimization.

       :raises NotImplementedError: Indicates that this method needs to be implemented in subclasses.

    .. method:: Module.save(*args, **kwargs)

       Saves the model to a file or database. Subclasses should implement this method to provide custom save functionality and may use the ``@explainable_gnn.decorator_save`` decorator to standardize save behavior.

       :raises NotImplementedError: Indicates that this method needs to be implemented in subclasses.

    .. method:: Module.load(*args, **kwargs)

       Loads the model from a file or database. This method should be implemented to reinitialize a model with previously saved states.

       :raises NotImplementedError: Indicates that this method needs to be implemented in subclasses.

    .. method:: Module.inference(*args, **kwargs)

       Transforms the model into an inference model, typically by optimizing for performance during the inference phase.

       :raises NotImplementedError: Indicates that this method needs to be implemented in subclasses.

    .. method:: Module.deploy(*args, **kwargs)

       Prepares the model for deployment, which may include integration into production environments or systems.

       :raises NotImplementedError: Indicates that this method needs to be implemented in subclasses.

    Examples
    --------
    Creating a custom model that inherits from ``Module`` and implements required methods:

    .. code-block:: python

        class CustomModel(Module):
            def parameters_calculation(self):
                # Custom parameter calculation logic
                pass

            def save(self):
                # Custom save logic
                pass

            def load(self):
                # Custom load logic
                pass

            def inference(self):
                # Custom inference preparation logic
                pass

            def deploy(self):
                # Custom deployment preparation logic
                pass

            def approximate(self):
                # Custom approximation logic
                pass

            def replace(self):
                # Custom model replacement logic
                pass

    Notes
    -----
    - The ``Module`` class is crucial for developers working within the Explainable GNN framework, ensuring that all models conform to the framework's operational standards and practices.
    - Implementers are encouraged to provide thorough implementations of the methods outlined, particularly those that handle the model’s lifecycle events such as saving, loading, and deploying.

    """

    def __init__(self, *args, **kwargs):
        super(Module, self).__init__()
        self.save_regular = False

        if kwargs.get("replace_module", None) is not None:
            self.replace_module = kwargs.get("replace_module")

    def replace(self, *args, **kwargs):
        """
        Replace the model
        """
        raise NotImplementedError

    def parameters_calculation(self):
        """
        It is required to translate process
        When the new module tries to replace the old module, the parameters may require to be recalculated
        """
        raise NotImplementedError

    def approximate(self, *args, **kwargs):
        """
        Approximate the model
        """
        raise NotImplementedError

    def save(self, *args, **kwargs):
        """
        Save the model
        When rewrite this function, please add the decorator @explainable_gnn.decorator_save
        """
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """
        Load the model
        """
        raise NotImplementedError

    def inference(self, *args, **kwargs):
        """
        Transform the model to an inference model
        """
        raise NotImplementedError

    def deploy(self, *args, **kwargs):
        """
        Transform the model to a deploy model
        """
        raise NotImplementedError


class TreeNode:
    """
    TreeNode Class
    ==============

    The ``TreeNode`` class represents a single node in a tree, holding references to its value, its parent, and its children.

    Constructor
    -----------
    .. method:: TreeNode.__init__(value, parent=None)

       Initializes a new instance of the ``TreeNode`` class.

       :param value: The value stored in the node.
       :param parent: A reference to the parent ``TreeNode`` object. Default is None, indicating that the node has no parent (root node).

    Methods
    -------
    .. method:: TreeNode.add_child(child)

       Adds a child node to this node's list of children.

       :param child: The ``TreeNode`` object to add as a child.

    .. method:: TreeNode.__str__()

       Returns a string representation of the node, showing its value.

    .. method:: TreeNode.__repr__()

       Returns a detailed string representation of the node for debugging purposes.

    """

    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.children = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def __str__(self):
        return f'Node({self.value})'

    def __repr__(self):
        return self.__str__()


class tree:
    """
    Tree Class
    ==========

    The ``Tree`` class manages a hierarchical tree structure, allowing for the construction and manipulation of a tree using nodes defined by the ``TreeNode`` class.

    Constructor
    -----------
    .. method:: Tree.__init__(edge_list=None)

       Initializes a new instance of the ``Tree`` class. Optionally builds the tree from a given edge list.

       :param edge_list: A list of tuples where each tuple (parent, child) represents an edge in the tree. The first element of the list is assumed to represent the root node.

    Methods
    -------
    .. method:: Tree.build_tree(edge_list)

       Builds the tree based on the provided edge list.

       :param edge_list: A list of tuples where each tuple (parent, child) represents an edge in the tree. The tree is built by connecting these nodes according to their parent-child relationships.

    .. method:: Tree.add_node(value, parent=None)

       Adds a new node to the tree.

       :param value: The value to store in the new node.
       :param parent: The parent ``TreeNode`` under which the new node will be added. If None, the new node will be set as the root of the tree.
       :return: The newly created ``TreeNode`` object.

       * Raises:
         - ValueError: If an attempt is made to set a root node when one already exists.

    .. method:: Tree.display(node=None, level=0)

       Recursively displays the tree structure starting from the node provided or from the root if no node is provided.

       :param node: The starting node for displaying the tree. If None, the root node is used.
       :param level: The initial indentation level for the display.

    .. method:: Tree.__str__()

       Returns a string representation of the tree by recursively displaying all nodes starting from the root.

    Examples
    --------
    Creating and displaying a tree:

    .. code-block:: python

        # Define an edge list for the tree
        edge_list = [(1, 2), (1, 3), (2, 4), (2, 5)]
        # Create a tree and build it using the edge list
        tree = Tree(edge_list)
        # Display the tree structure
        tree.display()

    """

    def __init__(self, edge_list=None):
        self.root = None
        if edge_list is not None:
            self.build_tree(edge_list)

    def build_tree(self, edge_list):
        self.root = TreeNode(edge_list[0][0])
        node_dict = {edge_list[0][0]: self.root}
        for parent, child in edge_list[1:]:
            if parent not in node_dict:
                node_dict[parent] = self.add_node(parent)
            if child not in node_dict:
                node_dict[child] = self.add_node(child, node_dict[parent])
            else:
                node_dict[child].parent = node_dict[parent]
                node_dict[parent].children.append(node_dict[child])

    def add_node(self, value, parent=None):
        new_node = TreeNode(value, parent)
        if parent is None:
            if self.root is None:
                self.root = new_node
            else:
                raise ValueError("Root already exists")
        else:
            parent.add_child(new_node)
        return new_node

    def display(self, node=None, level=0):
        if node is None:
            node = self.root
        print(' ' * level + str(node))
        for child in node.children:
            self.display(child, level + 4)

    def display_str(self, node=None, level=0):
        if node is None:
            node = self.root
        res = ' ' * level + str(node) + '\n'
        for child in node.children:
            res += self.display_str(child, level + 4)
        return res

    def __str__(self):
        return self.display_str(self.root)
