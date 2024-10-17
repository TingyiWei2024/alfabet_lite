from typing import Any, Callable, Dict, Hashable, Optional, Union
from abc import ABC, abstractmethod
import json
from inspect import getmembers

import numpy as np
import networkx as nx
import rdkit.Chem

import tensorflow as tf

from . import features
from .tokenizer import Tokenizer

"""
Ported Preprocessor class from nfp package.
"""

class Preprocessor(ABC):
    """A base class for graph preprocessing from the nfp package.

    Parameters
    ----------
    output_dtype
        A parameter used in child classes for determining the datatype of
        the returned arrays
    """

    def __init__(self, output_dtype: str = "int32"):
        self.output_dtype = output_dtype

    @abstractmethod
    def create_nx_graph(self, structure: Any, *args, **kwargs) -> nx.DiGraph:
        """Given an input structure object, convert it to a networkx digraph
        with node, edge, and graph features assigned.

        Parameters
        ----------
        structure
            Any input graph object
        kwargs
            keyword arguments passed from `__call__`, useful for specifying
            additional features in addition to the graph object.

        Returns
        -------
        nx.DiGraph
            A networkx graph with the node, edge, and graph features set
        """
        pass

    @abstractmethod
    def get_edge_features(
        self, edge_data: list, max_num_edges: int
    ) -> Dict[str, np.ndarray]:
        """Given a list of edge features from the nx.Graph, processes and
        concatenates them to an array.

        Parameters
        ----------
        edge_data
            A list of edge data generated by `nx_graph.edges(data=True)`
        max_num_edges
            If desired, this function should pad to a maximum number of edges
            passed from the `__call__` function.

        Returns
        -------
        Dict[str, np.ndarray]
            a dictionary of feature, array pairs, where array contains features for
            all edges in the graph.
        """
        pass

    @abstractmethod
    def get_node_features(
        self, node_data: list, max_num_nodes: int
    ) -> Dict[str, np.ndarray]:
        """Given a list of node features from the nx.Graph, processes and
        concatenates them to an array.

        Parameters
        ----------
        node_data
            A list of edge data generated by `nx_graph.nodes(data=True)`
        max_num_nodes
            If desired, this function should pad to a maximum number of nodes
            passed from the `__call__` function.

        Returns
        -------
        Dict[str, np.ndarray]
            a dictionary of feature, array pairs, where array contains features for
            all nodes in the graph.
        """
        pass

    @abstractmethod
    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        """Process the nx.graph features into a dictionary of arrays.

        Parameters
        ----------
        graph_data
            A dictionary of graph data generated by `nx_graph.graph`

        Returns
        -------
        Dict[str, np.ndarray]
            a dictionary of features for the graph
        """
        pass

    @staticmethod
    def get_connectivity(
        graph: nx.DiGraph, max_num_edges: int
    ) -> Dict[str, np.ndarray]:
        """Get the graph connectivity from the networkx graph

        Parameters
        ----------
        graph
            The input graph
        max_num_edges
            len(graph.edges), or the specified maximum number of graph edges

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary of with the single 'connectivity' key, containing an (n,2)
            array of (node_index, node_index) pairs indicating the start and end
            nodes for each edge.
        """
        connectivity = np.zeros((max_num_edges, 2), dtype="int64")
        if len(graph.edges) > 0:  # Handle odd case with no edges
            connectivity[: len(graph.edges)] = np.asarray(graph.edges)
        return {"connectivity": connectivity}

    def __call__(
        self,
        structure: Any,
        *args,
        train: bool = False,
        max_num_nodes: Optional[int] = None,
        max_num_edges: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Convert an input graph structure into a featurized set of node, edge,
         and graph-level features.

        Parameters
        ----------
        structure
            An input graph structure (i.e., molecule, crystal, etc.)
        train
            A training flag passed to `Tokenizer` member attributes
        max_num_nodes
            A size attribute passed to `get_node_features`, defaults to the
            number of nodes in the current graph
        max_num_edges
            A size attribute passed to `get_edge_features`, defaults to the
            number of edges in the current graph
        kwargs
            Additional features or parameters passed to `construct_nx_graph`

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary of key, array pairs as a single sample.
        """
        nx_graph = self.create_nx_graph(structure, *args, **kwargs)

        max_num_edges = len(nx_graph.edges) if max_num_edges is None else max_num_edges
        assert (
            len(nx_graph.edges) <= max_num_edges
        ), "max_num_edges too small for given input"

        max_num_nodes = len(nx_graph.nodes) if max_num_nodes is None else max_num_nodes
        assert (
            len(nx_graph.nodes) <= max_num_nodes
        ), "max_num_nodes too small for given input"

        # Make sure that Tokenizer classes are correctly initialized
        for _, tokenizer in getmembers(self, lambda x: type(x) == Tokenizer):
            tokenizer.train = train

        node_features = self.get_node_features(nx_graph.nodes(data=True), max_num_nodes)
        edge_features = self.get_edge_features(nx_graph.edges(data=True), max_num_edges)
        graph_features = self.get_graph_features(nx_graph.graph)
        connectivity = self.get_connectivity(nx_graph, max_num_edges)

        return {**node_features, **edge_features, **graph_features, **connectivity}

    def construct_feature_matrices(
        self, *args, train=False, **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        .. deprecated:: 0.3.0
            `construct_feature_matrices` will be removed in 0.4.0, use
            `__call__` instead
        """
        warnings.warn(
            "construct_feature_matrices is deprecated, use `call` instead as "
            "of nfp 0.4.0",
            DeprecationWarning,
        )
        return self(*args, train=train, **kwargs)

    def to_json(self, filename: str) -> None:
        """Serialize the classes's data to a json file"""
        with open(filename, "w") as f:
            json.dump(self, f, default=lambda x: x.__dict__)

    def from_json(self, filename: str) -> None:
        """Set's the class's data with attributes taken from the save file"""
        with open(filename, "r") as f:
            json_data = json.load(f)
        load_from_json(self, json_data)

    @property
    @abstractmethod
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        pass

    @property
    @abstractmethod
    def padding_values(self) -> Dict[str, tf.constant]:
        pass

    @property
    @abstractmethod
    def tfrecord_features(self) -> Dict[str, tf.io.FixedLenFeature]:
        pass

class MolPreprocessor(Preprocessor):
    def __init__(
        self,
        atom_features: Optional[Callable[[rdkit.Chem.Atom], Hashable]] = None,
        bond_features: Optional[Callable[[rdkit.Chem.Bond], Hashable]] = None,
        **kwargs,
    ) -> None:
        super(MolPreprocessor, self).__init__(**kwargs)

        self.atom_tokenizer = Tokenizer()
        self.bond_tokenizer = Tokenizer()

        if atom_features is None:
            atom_features = features.atom_features_v1

        if bond_features is None:
            bond_features = features.bond_features_v1

        self.atom_features = atom_features
        self.bond_features = bond_features

    def create_nx_graph(self, mol: rdkit.Chem.Mol, **kwargs) -> nx.DiGraph:
        g = nx.Graph(mol=mol)
        g.add_nodes_from(((atom.GetIdx(), {"atom": atom}) for atom in mol.GetAtoms()))
        g.add_edges_from(
            (
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), {"bond": bond})
                for bond in mol.GetBonds()
            )
        )
        return nx.DiGraph(g)

    def get_edge_features(
        self, edge_data: list, max_num_edges
    ) -> Dict[str, np.ndarray]:
        bond_feature_matrix = np.zeros(max_num_edges, dtype=self.output_dtype)
        for n, (start_atom, end_atom, bond_dict) in enumerate(edge_data):
            flipped = start_atom == bond_dict["bond"].GetEndAtomIdx()
            bond_feature_matrix[n] = self.bond_tokenizer(
                self.bond_features(bond_dict["bond"], flipped=flipped)
            )

        return {"bond": bond_feature_matrix}

    def get_node_features(
        self, node_data: list, max_num_nodes
    ) -> Dict[str, np.ndarray]:
        atom_feature_matrix = np.zeros(max_num_nodes, dtype=self.output_dtype)
        for n, atom_dict in node_data:
            atom_feature_matrix[n] = self.atom_tokenizer(
                self.atom_features(atom_dict["atom"])
            )
        return {"atom": atom_feature_matrix}

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        return {}

    @property
    def atom_classes(self) -> int:
        """The number of atom types found (includes the 0 null-atom type)"""
        return self.atom_tokenizer.num_classes + 1

    @property
    def bond_classes(self) -> int:
        """The number of bond types found (includes the 0 null-bond type)"""
        return self.bond_tokenizer.num_classes + 1

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        if tf is None:
            raise ImportError("Tensorflow was not found")
        return {
            "atom": tf.TensorSpec(shape=(None,), dtype=self.output_dtype),
            "bond": tf.TensorSpec(shape=(None,), dtype=self.output_dtype),
            "connectivity": tf.TensorSpec(shape=(None, 2), dtype=self.output_dtype),
        }

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        """Defaults to zero for each output"""
        if tf is None:
            raise ImportError("Tensorflow was not found")
        return {
            key: tf.constant(0, dtype=self.output_dtype)
            for key in self.output_signature.keys()
        }

    @property
    def tfrecord_features(self) -> Dict[str, tf.io.FixedLenFeature]:
        """For loading preprocessed inputs from a tf records file"""
        if tf is None:
            raise ImportError("Tensorflow was not found")
        return {
            key: tf.io.FixedLenFeature(
                [], dtype=self.output_dtype if len(val.shape) == 0 else tf.string
            )
            for key, val in self.output_signature.items()
        }


class SmilesPreprocessor(MolPreprocessor):
    def __init__(self, *args, explicit_hs: bool = True, **kwargs):
        super(SmilesPreprocessor, self).__init__(*args, **kwargs)
        self.explicit_hs = explicit_hs

    def create_nx_graph(self, smiles: str, *args, **kwargs) -> nx.DiGraph:
        mol = rdkit.Chem.MolFromSmiles(smiles)
        if self.explicit_hs:
            mol = rdkit.Chem.AddHs(mol)
        return super(SmilesPreprocessor, self).create_nx_graph(mol, *args, **kwargs)

class BondIndexPreprocessor(MolPreprocessor):
    def get_edge_features(
        self, edge_data: list, max_num_edges
    ) -> Dict[str, np.ndarray]:
        bond_indices = np.zeros(max_num_edges, dtype=self.output_dtype)
        for n, (_, _, edge_dict) in enumerate(edge_data):
            bond_indices[n] = edge_dict["bond"].GetIdx()
        edge_features = super(BondIndexPreprocessor, self).get_edge_features(
            edge_data, max_num_edges
        )
        return {"bond_indices": bond_indices, **edge_features}

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        if tf is None:
            raise ImportError("Tensorflow was not found")

        signature = super(BondIndexPreprocessor, self).output_signature
        signature["bond_indices"] = tf.TensorSpec(
            shape=(None,), dtype=self.output_dtype
        )
        return signature

class SmilesBondIndexPreprocessor(SmilesPreprocessor, BondIndexPreprocessor):
    pass

def load_from_json(obj: Any, data: Dict):
    """Function to set member attributes from json data recursively.

    Parameters
    ----------
    obj
        the class to initialize
    data
        a dictionary of potentially nested attribute - value pairs

    Returns
    -------
    Any
        The object, with attributes set to those from the data file.
    """

    for key, val in obj.__dict__.items():
        try:
            if isinstance(val, type(data[key])):
                obj.__dict__[key] = data[key]
            elif hasattr(val, "__dict__"):
                load_from_json(val, data[key])

        except KeyError:
            logger.warning(
                f"{key} not found in JSON file, it may have been created with"
                " an older nfp version"
            )