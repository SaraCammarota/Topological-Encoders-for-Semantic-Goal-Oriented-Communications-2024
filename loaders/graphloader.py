
import torch
from torch.utils.data import Subset
import numpy as np
import json
from abc import ABC
import os
from torch.utils.data import Dataset
import torch_geometric
from sklearn.model_selection import StratifiedKFold
import omegaconf
import hashlib
from torch_geometric.io import fs
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
import torch_geometric
from collections import defaultdict

from typing import Any

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class DataloadDataset(torch_geometric.data.Dataset):
    """Custom dataset to return all the values added to the dataset object.

    Parameters
    ----------
    data_lst : list[torch_geometric.data.Data]
        List of torch_geometric.data.Data objects.
    """

    def __init__(self, data_lst):
        super().__init__()
        self.data_lst = data_lst

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self.data_lst)})"

    def get(self, idx):
        """Get data object from data list.

        Parameters
        ----------
        idx : int
            Index of the data object to get.

        Returns
        -------
        tuple
            Tuple containing a list of all the values for the data and the corresponding keys.
        """
        data = self.data_lst[idx]
        keys = list(data.keys())
        return ([data[key] for key in keys], keys)

    def len(self):
        """Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data_lst)


PLANETOID_DATASETS = [
    "Cora",
    "citeseer",
    "PubMed",
]

TU_DATASETS = [
    "MUTAG",
    "ENZYMES",
    "PROTEINS",
    "COLLAB",
    "IMDB-BINARY",
    "IMDB-MULTI",
    "REDDIT-BINARY",
    "NCI1",
    "NCI109",
]

FIXED_SPLITS_DATASETS = ["ZINC", "AQSOL"]

HETEROPHILIC_DATASETS = [
    "amazon_ratings",
    "questions",
    "minesweeper",
    "roman_empire",
    "tolokers",
]

PYG_DATASETS = (
    PLANETOID_DATASETS
    + TU_DATASETS
    + FIXED_SPLITS_DATASETS
    + HETEROPHILIC_DATASETS
)




class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

"""Abstract Loader class."""

from abc import ABC, abstractmethod

import torch_geometric
from omegaconf import DictConfig


class AbstractLoader(ABC):
    """Abstract class that provides an interface to load data.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        self.cfg = parameters

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.cfg})"

    @abstractmethod
    def load(self) -> torch_geometric.data.Data:
        """Load data into Data.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError



class GraphLoader(AbstractLoader):
    """Loader for graph datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    **kwargs : dict
        Additional keyword arguments.

    Notes
    -----
    The parameters must contain the following keys:
    - data_dir (str): The directory where the dataset is stored.
    - data_name (str): The name of the dataset.
    - data_type (str): The type of the dataset.
    - split_type (str): The type of split to be used. It can be "fixed", "random", or "k-fold".
    If split_type is "random", the parameters must also contain the following keys:
    - data_seed (int): The seed for the split.
    - data_split_dir (str): The directory where the split is stored.
    - train_prop (float): The proportion of the training set.
    If split_type is "k-fold", the parameters must also contain the following keys:
    - data_split_dir (str): The directory where the split is stored.
    - k (int): The number of folds.
    - data_seed (int): The seed for the split.
    The parameters can be defined in a yaml file and then loaded using `omegaconf.OmegaConf.load('path/to/dataset/config.yaml')`.
    """

    def __init__(self, parameters: DictConfig, **kwargs):
        super().__init__(parameters)
        self.parameters = parameters

    def __repr__(self) -> str:
        """Return a string representation of the GraphLoader object.

        Returns
        -------
        str
            String representation of the GraphLoader object.
        """
        return f"{self.__class__.__name__}(parameters={self.parameters})"

    def load(self) -> tuple[torch_geometric.data.Dataset, str]:
        """Load graph dataset.

        Returns
        -------
        tuple[torch_geometric.data.Dataset, str]
            Tuple containing the loaded data and the data directory.
        """
        # Define the path to the data directory
        root_data_dir = self.parameters.data_dir
        data_dir = os.path.join(root_data_dir, self.parameters.data_name)
        if (
            self.parameters.data_name in PLANETOID_DATASETS
            and self.parameters.data_type == "cocitation"
        ):
            dataset = torch_geometric.datasets.Planetoid(
                root=root_data_dir,
                name=self.parameters.data_name,
            )

        elif self.parameters.data_name in TU_DATASETS:

            dataset = torch_geometric.datasets.TUDataset(
                root=root_data_dir,
                name=self.parameters.data_name,
                use_node_attr=False,
            )

        elif self.parameters.data_name in FIXED_SPLITS_DATASETS:
            datasets = []
            for split in ["train", "val", "test"]:
                if self.parameters.data_name == "ZINC":
                    datasets.append(
                        torch_geometric.datasets.ZINC(
                            root=root_data_dir,
                            subset=True,
                            split=split,
                        )
                    )
                elif self.parameters.data_name == "AQSOL":
                    datasets.append(
                        torch_geometric.datasets.AQSOL(
                            root=root_data_dir,
                            split=split,
                        )
                    )
            # The splits are predefined
            # Extract and prepare split_idx
            split_idx = {"train": np.arange(len(datasets[0]))}
            split_idx["valid"] = np.arange(
                len(datasets[0]), len(datasets[0]) + len(datasets[1])
            )
            split_idx["test"] = np.arange(
                len(datasets[0]) + len(datasets[1]),
                len(datasets[0]) + len(datasets[1]) + len(datasets[2]),
            )
            # Join dataset to process it
            dataset = datasets[0] + datasets[1] + datasets[2]
            dataset.split_idx = split_idx
            data_dir = root_data_dir

        elif self.parameters.data_name in HETEROPHILIC_DATASETS:
            dataset = torch_geometric.datasets.HeterophilousGraphDataset(
                root=root_data_dir,
                name=self.parameters.data_name,
            )

        # elif self.parameters.data_name in ["US-county-demos"]:
        #     dataset = USCountyDemosDataset(
        #         root=root_data_dir,
        #         name=self.parameters["data_name"],
        #         parameters=self.parameters,
        #     )
        #     # Need to redefine data_dir for the (year, task_variable) pair chosen
        #     data_dir = dataset.processed_root

        # elif self.parameters.data_name in ["manual"]:
        #     data = load_manual_graph()
        #     dataset = DataloadDataset([data], data_dir)

        else:
            raise NotImplementedError(
                f"Dataset {self.parameters.data_name} not implemented"
            )

        return dataset, data_dir
    



def ensure_serializable(obj):
    """Ensure that the object is serializable.

    Parameters
    ----------
    obj : object
        Object to ensure serializability.

    Returns
    -------
    object
        Object that is serializable.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = ensure_serializable(value)
        return obj
    elif isinstance(obj, list | tuple):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return {ensure_serializable(item) for item in obj}
    elif isinstance(obj, str | int | float | bool | type(None)):
        return obj
    elif isinstance(obj, omegaconf.dictconfig.DictConfig):
        return dict(obj)
    else:
        return None



# Generate splits in different fasions
def k_fold_split(labels, parameters):
    """Return train and valid indices as in K-Fold Cross-Validation.

    If the split already exists it loads it automatically, otherwise it creates the
    split file for the subsequent runs.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor.
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    dict
        Dictionary containing the train, validation and test indices, with keys "train", "valid", and "test".
    """

    data_dir = parameters.data_split_dir
    k = parameters.k
    fold = parameters.data_seed
    assert fold < k, "data_seed needs to be less than k"

    torch.manual_seed(0)
    np.random.seed(0)

    split_dir = os.path.join(data_dir, f"{k}-fold")

    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)

    split_path = os.path.join(split_dir, f"{fold}.npz")
    if not os.path.isfile(split_path):
        n = labels.shape[0]
        x_idx = np.arange(n)
        x_idx = np.random.permutation(x_idx)
        labels = labels[x_idx]

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        for fold_n, (train_idx, valid_idx) in enumerate(
            skf.split(x_idx, labels)
        ):
            split_idx = {
                "train": train_idx,
                "valid": valid_idx,
                "test": valid_idx,
            }

            # Check that all nodes/graph have been assigned to some split
            assert np.all(
                np.sort(
                    np.array(
                        split_idx["train"].tolist()
                        + split_idx["valid"].tolist()
                    )
                )
                == np.sort(np.arange(len(labels)))
            ), "Not every sample has been loaded."
            split_path = os.path.join(split_dir, f"{fold_n}.npz")

            np.savez(split_path, **split_idx)

    split_path = os.path.join(split_dir, f"{fold}.npz")
    split_idx = np.load(split_path)

    # Check that all nodes/graph have been assigned to some split
    assert (
        np.unique(
            np.array(
                split_idx["train"].tolist()
                + split_idx["valid"].tolist()
                + split_idx["test"].tolist()
            )
        ).shape[0]
        == labels.shape[0]
    ), "Not all nodes within splits"

    return split_idx


def random_splitting(labels, parameters, global_data_seed=42):
    r"""Randomly splits label into train/valid/test splits.

    Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor.
    parameters : DictConfig
        Configuration parameter.
    global_data_seed : int
        Seed for the random number generator.

    Returns
    -------
    dict:
        Dictionary containing the train, validation and test indices with keys "train", "valid", and "test".
    """
    fold = parameters.data_seed
    data_dir = parameters.data_split_dir
    train_prop = parameters.train_prop
    valid_prop = (1 - train_prop) / 2

    # Create split directory if it does not exist
    split_dir = os.path.join(
        data_dir, f"train_prop={train_prop}_global_seed={global_data_seed}"
    )
    generate_splits = False
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
        generate_splits = True

    # Generate splits if they do not exist
    if generate_splits:
        # Set initial seed
        torch.manual_seed(global_data_seed)
        np.random.seed(global_data_seed)
        # Generate a split
        n = labels.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        # Generate 10 splits
        for fold_n in range(10):
            # Permute indices
            perm = torch.as_tensor(np.random.permutation(n))

            train_indices = perm[:train_num]
            val_indices = perm[train_num : train_num + valid_num]
            test_indices = perm[train_num + valid_num :]
            split_idx = {
                "train": train_indices,
                "valid": val_indices,
                "test": test_indices,
            }

            # Save generated split
            split_path = os.path.join(split_dir, f"{fold_n}.npz")
            np.savez(split_path, **split_idx)

    # Load the split
    split_path = os.path.join(split_dir, f"{fold}.npz")
    split_idx = np.load(split_path)

    # Check that all nodes/graph have been assigned to some split
    assert (
        np.unique(
            np.array(
                split_idx["train"].tolist()
                + split_idx["valid"].tolist()
                + split_idx["test"].tolist()
            )
        ).shape[0]
        == labels.shape[0]
    ), "Not all nodes within splits"

    return split_idx


def assing_train_val_test_mask_to_graphs(dataset, split_idx):
    r"""Split the graph dataset into train, validation, and test datasets.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Considered dataset.
    split_idx : dict
        Dictionary containing the train, validation, and test indices.

    Returns
    -------
    list:
        List containing the train, validation, and test datasets.
    """
    data_train_lst, data_val_lst, data_test_lst = [], [], []

    # Go over each of the graph and assign correct label
    for i in range(len(dataset)):
        graph = dataset[i]
        assigned = False
        if i in split_idx["train"]:
            graph.train_mask = torch.Tensor([1]).long()
            graph.val_mask = torch.Tensor([0]).long()
            graph.test_mask = torch.Tensor([0]).long()
            data_train_lst.append(graph)
            assigned = True

        if i in split_idx["valid"]:
            graph.train_mask = torch.Tensor([0]).long()
            graph.val_mask = torch.Tensor([1]).long()
            graph.test_mask = torch.Tensor([0]).long()
            data_val_lst.append(graph)
            assigned = True

        if i in split_idx["test"]:
            graph.train_mask = torch.Tensor([0]).long()
            graph.val_mask = torch.Tensor([0]).long()
            graph.test_mask = torch.Tensor([1]).long()
            data_test_lst.append(graph)
            assigned = True
        if not assigned:
            raise ValueError("Graph not in any split")

    return (
        DataloadDataset(data_train_lst),
        DataloadDataset(data_val_lst),
        DataloadDataset(data_test_lst),
    )


def load_transductive_splits(dataset, parameters):
    r"""Load the graph dataset with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    list:
        List containing the train, validation, and test splits.
    """
    # Extract labels from dataset object
    assert (
        len(dataset) == 1
    ), "Dataset should have only one graph in a transductive setting."

    data = dataset.data_list[0]
    labels = data.y.numpy()

    # Ensure labels are one dimensional array
    assert len(labels.shape) == 1, "Labels should be one dimensional array"

    if parameters.split_type == "random":
        splits = random_splitting(labels, parameters)

    elif parameters.split_type == "k-fold":
        splits = k_fold_split(labels, parameters)

    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose either 'random' or 'k-fold'"
        )

    # Assign train val test masks to the graph
    data.train_mask = torch.from_numpy(splits["train"])
    data.val_mask = torch.from_numpy(splits["valid"])
    data.test_mask = torch.from_numpy(splits["test"])

    if parameters.get("standardize", False):
        # Standardize the node features respecting train mask
        data.x = (data.x - data.x[data.train_mask].mean(0)) / data.x[
            data.train_mask
        ].std(0)
        data.y = (data.y - data.y[data.train_mask].mean(0)) / data.y[
            data.train_mask
        ].std(0)

    return DataloadDataset([data]), None, None


def load_inductive_splits(dataset, parameters):
    r"""Load multiple-graph datasets with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    list:
        List containing the train, validation, and test splits.
    """
    # Extract labels from dataset object
    assert (
        len(dataset) > 1
    ), "Datasets should have more than one graph in an inductive setting."
    labels = np.array(
        [data.y.squeeze(0).numpy() for data in dataset.data_list]
    )

    if parameters.split_type == "random":
        split_idx = random_splitting(labels, parameters)

    elif parameters.split_type == "k-fold":
        split_idx = k_fold_split(labels, parameters)

    elif parameters.split_type == "fixed" and hasattr(dataset, "split_idx"):
        split_idx = dataset.split_idx

    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose either 'random', 'k-fold' or 'fixed'.\
            If 'fixed' is chosen, the dataset should have the attribute split_idx"
        )

    train_dataset, val_dataset, test_dataset = (
        assing_train_val_test_mask_to_graphs(dataset, split_idx)
    )

    return train_dataset, val_dataset, test_dataset


def load_coauthorship_hypergraph_splits(data, parameters, train_prop=0.5):
    r"""Load the split generated by rand_train_test_idx function.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.
    train_prop : float
        Proportion of training data.

    Returns
    -------
    torch_geometric.data.Data:
        Graph dataset with the specified split.
    """

    data_dir = os.path.join(
        parameters.data_split_dir, f"train_prop={train_prop}"
    )
    load_path = f"{data_dir}/split_{parameters.data_seed}.npz"
    splits = np.load(load_path, allow_pickle=True)

    # Upload masks
    data.train_mask = torch.from_numpy(splits["train"])
    data.val_mask = torch.from_numpy(splits["valid"])
    data.test_mask = torch.from_numpy(splits["test"])

    # Check that all nodes assigned to splits
    assert (
        torch.unique(
            torch.concat([data.train_mask, data.val_mask, data.test_mask])
        ).shape[0]
        == data.num_nodes
    ), "Not all nodes within splits"
    return DataloadDataset([data]), None, None


class PreProcessor(torch_geometric.data.InMemoryDataset):
    """Preprocessor for datasets.

    Parameters
    ----------
    dataset : list
        List of data objects.
    data_dir : str
        Path to the directory containing the data.
    transforms_config : DictConfig, optional
        Configuration parameters for the transforms (default: None).
    **kwargs : optional
        Optional additional arguments.
    """

    def __init__(self, dataset, data_dir, transforms_config=None, **kwargs):
        if isinstance(dataset, torch_geometric.data.Dataset):
            data_list = [dataset.get(idx) for idx in range(len(dataset))]
        elif isinstance(dataset, torch.utils.data.Dataset):
            data_list = [dataset[idx] for idx in range(len(dataset))]
        elif isinstance(dataset, torch_geometric.data.Data):
            data_list = [dataset]
        self.data_list = data_list
        self.transforms_applied = False
        super().__init__(data_dir, None, None, **kwargs)
        self.load(data_dir + "/processed/data.pt")

        self.data_list = [self.get(idx) for idx in range(len(self))]
        # Some datasets have fixed splits, and those are stored as split_idx during loading
        # We need to store this information to be able to reproduce the splits afterwards
        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        if self.transforms_applied:
            return self.root
        else:
            return self.root + "/processed"

    @property
    def processed_file_names(self) -> str:
        """Return the name of the processed file.

        Returns
        -------
        str
            Name of the processed file.
        """
        return "data.pt"

    def set_processed_data_dir(
        self, pre_transforms_dict, data_dir, transforms_config
    ) -> None:
        """Set the processed data directory.

        Parameters
        ----------
        pre_transforms_dict : dict
            Dictionary containing the pre-transforms.
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.
        """
        # Use self.transform_parameters to define unique save/load path for each transform parameters
        repo_name = "_".join(list(transforms_config.keys()))
        transforms_parameters = {
            transform_name: transform.parameters
            for transform_name, transform in pre_transforms_dict.items()
        }
        params_hash = make_hash(transforms_parameters)
        self.transforms_parameters = ensure_serializable(transforms_parameters)
        self.processed_data_dir = os.path.join(
            *[data_dir, repo_name, f"{params_hash}"]
        )

    def save_transform_parameters(self) -> None:
        """Save the transform parameters."""
        # Check if root/params_dict.json exists, if not, save it
        path_transform_parameters = os.path.join(
            self.processed_data_dir, "path_transform_parameters_dict.json"
        )
        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(self.transforms_parameters, f, indent=4)
        else:
            # If path_transform_parameters exists, check if the transform_parameters are the same
            with open(path_transform_parameters) as f:
                saved_transform_parameters = json.load(f)

            if saved_transform_parameters != self.transforms_parameters:
                raise ValueError(
                    "Different transform parameters for the same data_dir"
                )

            print(
                f"Transform parameters are the same, using existing data_dir: {self.processed_data_dir}"
            )

    def process(self) -> None:
        """Method that processes the data."""
        self.data_list = (
            [self.pre_transform(d) for d in self.data_list]
            if self.pre_transform is not None
            else self.data_list
        )

        self._data, self.slices = self.collate(self.data_list)
        self._data_list = None  # Reset cache.

        assert isinstance(self._data, torch_geometric.data.Data)
        self.save(self.data_list, self.processed_paths[0])

    def load(self, path: str) -> None:
        r"""Load the dataset from the file path `path`.

        Parameters
        ----------
        path : str
            The path to the processed data.
        """
        out = fs.torch_load(path)
        assert isinstance(out, tuple)
        assert len(out) >= 2 and len(out) <= 4
        if len(out) == 2:  # Backward compatibility (1).
            data, self.slices = out
        elif len(out) == 3:  # Backward compatibility (2).
            data, self.slices, data_cls = out
        else:  # TU Datasets store additional element (__class__) in the processed file
            data, self.slices, sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    def load_dataset_splits(
        self, split_params
    ) -> tuple[
        DataloadDataset, DataloadDataset | None, DataloadDataset | None
    ]:
        """Load the dataset splits.

        Parameters
        ----------
        split_params : dict
            Parameters for loading the dataset splits.

        Returns
        -------
        tuple
            A tuple containing the train, validation, and test datasets.
        """
        if not split_params.get("learning_setting", False):
            raise ValueError("No learning setting specified in split_params")

        if split_params.learning_setting == "inductive":
            return load_inductive_splits(self, split_params)
        elif split_params.learning_setting == "transductive":
            return load_transductive_splits(self, split_params)
        else:
            raise ValueError(
                f"Invalid '{split_params.learning_setting}' learning setting.\
                Please define either 'inductive' or 'transductive'."
            )



# class GraphLoader(AbstractLoader):
#     def __init__(self, parameters):
#         super().__init__()
#         self.parameters = parameters
#         # Still not instantiated

#     def load(self):
#         data_dir = os.path.join(
#             self.parameters["data_dir"], self.parameters["data_name"]
#         )



# # could add QM9 dataset https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html

#         if self.parameters["data_name"] in [
#             "MUTAG",
#             "ENZYMES",
#             "PROTEINS",
#             #"COLLAB",
#             #"IMDB-BINARY",
#             #"IMDB-MULTI",
#             "REDDIT-BINARY",
#             "NCI1",
#             "NCI109",

#         ]:


#             dataset = torch_geometric.datasets.TUDataset(
#                 root=self.parameters["data_dir"],
#                 name=self.parameters["data_name"],
#                 use_node_attr=False,
                
#             )


#             dataset = load_graph_tudataset_split(dataset, self.parameters)

#             return dataset

#         elif self.parameters["data_name"] in ["CORA"]:

#             dataset = torch_geometric.datasets.Planetoid(
#                 root=self.parameters["data_dir"],
#                 name=self.parameters["data_name"],
                
#             )[0]

#             split_idx = {"train": np.array(np.where(dataset.train_mask.bool()==1))[0]}
#             split_idx["validation"] = np.array(np.where(dataset.val_mask.bool()==1))[0]
#             split_idx["test"] = np.array(np.where(dataset.test_mask.bool()==1))[0]



#         elif self.parameters["data_name"] in ["ZINC"]:
#             datasets = []
#             for split in ["train", "val", "test"]:
#                 datasets.append(
#                     torch_geometric.datasets.ZINC(
#                         root=self.parameters["data_dir"],
#                         subset=True,
#                         split=split,
#                     )
#                 )

#             assert self.parameters["split_type"] == "fixed"
#             # The splits are predefined
#             # Extract and prepare split_idx
#             split_idx = {"train": np.arange(len(datasets[0]))}

#             split_idx["valid"] = np.arange(
#                 len(datasets[0]), len(datasets[0]) + len(datasets[1])
#             )

#             split_idx["test"] = np.arange(
#                 len(datasets[0]) + len(datasets[1]),
#                 len(datasets[0]) + len(datasets[1]) + len(datasets[2]),
#             )

#             # Join dataset to process it
#             joined_dataset = datasets[0] + datasets[1] + datasets[2]

#             # if self.transforms_config is not None:
#             #     joined_dataset = Preprocessor(
#             #         data_dir,
#             #         joined_dataset,
#             #         self.transforms_config,
#             #     )

#             # Split back the into train/val/test datasets
#             dataset = get_train_val_test_graph_datasets(joined_dataset, split_idx)

#         elif self.parameters["data_name"] in ["AQSOL"]:
#             datasets = []
#             for split in ["train", "val", "test"]:
#                 datasets.append(
#                     torch_geometric.datasets.AQSOL(
#                         root=self.parameters["data_dir"],
#                         split=split,
#                     )
#                 )
#             # The splits are predefined
#             # Extract and prepare split_idx
#             split_idx = {"train": np.arange(len(datasets[0]))}

#             split_idx["valid"] = np.arange(
#                 len(datasets[0]), len(datasets[0]) + len(datasets[1])
#             )

#             split_idx["test"] = np.arange(
#                 len(datasets[0]) + len(datasets[1]),
#                 len(datasets[0]) + len(datasets[1]) + len(datasets[2]),
#             )

#             # Join dataset to process it
#             joined_dataset = datasets[0] + datasets[1] + datasets[2]

#             # if self.transforms_config is not None:
#             #     joined_dataset = Preprocessor(
#             #         data_dir,
#             #         joined_dataset,
#             #         self.transforms_config,
#             #     )

#             # Split back the into train/val/test datasets
#             dataset = get_train_val_test_graph_datasets(joined_dataset, split_idx)
#         else:
#             raise NotImplementedError(
#                 f"Dataset {self.parameters['data_name']} not implemented"
#             )

#         return dataset

def k_fold_split(dataset, parameters, test_ratio = 0.2, ignore_negative=True):
    """
    Returns train and valid indices as in K-Fold Cross-Validation. If the split already exists it loads it automatically, otherwise it creates the split file for the subsequent runs.

    :param dataset: Dataset object containing either one or multiple graphs
    :param data_dir: The directory where the data is stored, it will be used to store the splits
    :param parameters: DictConfig containing the parameters for the dataset
    :param ignore_negative: If True the function ignores negative labels. Default True.
    :return split_idx: A dictionary containing "train" and "valid" tensors with the respective indices.
    """
    data_dir = parameters.data_split_dir
    k = parameters.k
    fold = parameters.data_seed
    assert fold < k, "data_seed needs to be less than k"

    torch.manual_seed(0)
    np.random.seed(0)

    split_dir = os.path.join(data_dir, f"{k}-fold")
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
    split_path = os.path.join(split_dir, f"{fold}.npz")
    if os.path.isfile(split_path):
        split_idx = np.load(split_path)
        return split_idx
    else:
        if parameters.task_level == "graph":
            labels = dataset.y

        # need to look into this more
        else:
            if len(dataset) == 1:
                labels = dataset.y
            else:
                # This is the case of node level task with multiple graphs
                # Here dataset.y cannot be used to measure the number of elements to associate to the splits
                labels = torch.ones(len(dataset))

        if ignore_negative:
            labeled_nodes = torch.where(labels != -1)[0]
        else:
            labeled_nodes = labels

        n = labeled_nodes.shape[0]

        #when is the len == 1?
        if len(dataset) == 1:
            y = dataset[0].y.squeeze(0).numpy()
        else:
            y = np.array([data.y.squeeze(0).numpy() for data in dataset])

        x_idx = np.arange(n)
        x_idx = np.random.permutation(x_idx)
        y = y[x_idx]

        #Set aside test set
        num_test = int(test_ratio*n)
        x_idx_test = x_idx[:num_test]

        y_train_val = y[num_test:]

        #y_test = y[test_ratio*n:]
        x_train_val = x_idx[num_test:]


        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        for fold_n, (train_idx, valid_idx) in enumerate(skf.split(x_train_val, y_train_val)):

            # Convert relative indices to original indices
            train_indices = x_train_val[train_idx]
            valid_indices = x_train_val[valid_idx]

            split_idx = {"train": train_indices, "valid": valid_indices, "test": x_idx_test}

            assert np.all(np.sort(np.concatenate([train_indices, valid_indices])) == np.sort(x_train_val)), "Not every sample has been loaded."

            split_path = os.path.join(split_dir, f"{fold_n}.npz")
            np.savez(split_path, **split_idx)

        return split_idx



def load_graph_tudataset_split(dataset, cfg):


    '''
    Supports 'test' or 'k-fold' split types.
    :param dataset: Graph dataset to be split.
    :param cfg: Configuration dictionary specifying the type of split and other parameters.
    :return: A list containing train, validation, and test datasets.
    '''

    if cfg["split_type"] == "test":
        labels = dataset.y
        split_idx = rand_train_test_val_idx(labels)

    elif cfg["split_type"] == "k-fold":
        split_idx = k_fold_split(dataset, cfg)
    else:
        raise NotImplementedError(
            f"split_type {cfg['split_type']} not valid. Choose either 'test' or 'k-fold'"
        )

    train_dataset, val_dataset, test_dataset = get_train_val_test_graph_datasets(
        dataset, split_idx
    )

    return [train_dataset, val_dataset, test_dataset]

def get_train_val_test_graph_datasets(dataset, split_idx):
    '''
    Segregates the graphs into training, validation, and test datasets based on the indices provided in `split_idx`.
    :param dataset: Complete graph dataset to be divided.
    :param split_idx: Dictionary containing indices for training, validation, and testing splits.
    :return: Three datasets corresponding to the training, validation, and test splits.
    '''
    final_list = []

    # Go over each of the graph and assign correct label
    for i in range(len(dataset)):
        graph = dataset[i]
        assigned = False
        if i in split_idx["train"]:
            graph.train_mask = torch.Tensor([1]).long()
            graph.val_mask = torch.Tensor([0]).long()
            graph.test_mask = torch.Tensor([0]).long()
            #dataset[i] = graph
            final_list.append(graph)
            assigned = True

        if i in split_idx["valid"]:
            graph.train_mask = torch.Tensor([0]).long()
            graph.val_mask = torch.Tensor([1]).long()
            graph.test_mask = torch.Tensor([0]).long()
            #dataset[i] = graph
            final_list.append(graph)
            assigned = True

        if i in split_idx["test"]:
            graph.train_mask = torch.Tensor([0]).long()
            graph.val_mask = torch.Tensor([0]).long()
            graph.test_mask = torch.Tensor([1]).long()
            #dataset[i] = graph
            final_list.append(graph)
            assigned = True
        if not assigned:
            raise ValueError("Graph not in any split")

    new_data = CustomDataset(final_list)


    datasets = [Subset(new_data, split_idx["train"]), Subset(new_data, split_idx["valid"]), Subset(new_data, split_idx["test"])]


    return datasets



def rand_train_test_val_idx(labels, train_ratio=0.8):
    """
    Generates random indices for training, testing, and validation splits from a set of labels, with a specified training ratio.
    :param labels: Array or list of labels associated with the dataset elements.
    :param train_ratio: Proportion of the dataset to include in the training set (default is 0.8).
    :return: A dictionary containing indices for the training, validation, and test splits.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    num_samples = len(labels)
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    train_size = int(train_ratio * num_samples)
    test_valid_size = int((num_samples - train_size)/2)

    train_idx = idx[:train_size]
    valid_idx = idx[train_size:(train_size+test_valid_size)]
    test_idx = idx[(train_size+test_valid_size):]

    split_idx = {"train": train_idx, "valid": valid_idx,  "test": test_idx}

    return split_idx



def make_hash(o):
    """Make a hash from a dictionary, list, tuple or set to any level, that contains only other hashable types.

    Parameters
    ----------
    o : dict, list, tuple, set
        Object to hash.

    Returns
    -------
    int
        Hash of the object.
    """
    sha1 = hashlib.sha1()
    sha1.update(str.encode(str(o)))
    hash_as_hex = sha1.hexdigest()
    # Convert the hex back to int and restrict it to the relevant int range
    return int(hash_as_hex, 16) % 4294967295









class TBXDataloader(LightningDataModule):
    r"""This class takes care of returning the dataloaders for the training, validation, and test datasets.

    It also handles the collate function. The class is designed to work with the `torch` dataloaders.

    Parameters
    ----------
    dataset_train : DataloadDataset
        The training dataset.
    dataset_val : DataloadDataset, optional
        The validation dataset (default: None).
    dataset_test : DataloadDataset, optional
        The test dataset (default: None).
    batch_size : int, optional
        The batch size for the dataloader (default: 1).
    num_workers : int, optional
        The number of worker processes to use for data loading (default: 0).
    pin_memory : bool, optional
        If True, the data loader will copy tensors into pinned memory before returning them (default: False).
    **kwargs : optional
        Additional arguments.

    References
    ----------
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset_train: DataloadDataset,
        dataset_val: DataloadDataset = None,
        dataset_test: DataloadDataset = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["dataset_train", "dataset_val", "dataset_test"],
        )
        self.dataset_train = dataset_train
        self.batch_size = batch_size

        if dataset_val is None and dataset_test is None:
            # Transductive setting
            self.dataset_val = dataset_train
            self.dataset_test = dataset_train
            assert (
                self.batch_size == 1
            ), "Batch size must be 1 for transductive setting."
        else:
            self.dataset_val = dataset_val
            self.dataset_test = dataset_test
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = kwargs.get("persistent_workers", False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset_train={self.dataset_train}, dataset_val={self.dataset_val}, dataset_test={self.dataset_test}, batch_size={self.batch_size})"

    def train_dataloader(self) -> DataLoader:
        r"""Create and return the train dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The train dataloader.
        """
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        r"""Create and return the validation dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The validation dataloader.
        """
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        r"""Create and return the test dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The test dataloader.
        """
        if self.dataset_test is None:
            raise ValueError("There is no test dataloader.")
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def teardown(self, stage: str | None = None) -> None:
        r"""Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and `trainer.predict()`.

        Parameters
        ----------
        stage : str, optional
            The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"` (default: None).
        """

    def state_dict(self) -> dict[Any, Any]:
        r"""Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns
        -------
        dict
            A dictionary containing the datamodule state that you want to save.
        """
        return {}



class DomainData(torch_geometric.data.Data):
    r"""Helper Data class so that not only sparse matrices with adj in the name can work with PyG dataloaders.

    It overwrites some methods from `torch_geometric.data.Data`
    """

    def is_valid(self, string):
        r"""Check if the string contains any of the valid names.

        Parameters
        ----------
        string : str
            String to check.

        Returns
        -------
        bool
            Whether the string contains any of the valid names.
        """
        valid_names = ["adj", "incidence", "laplacian"]
        return any(name in string for name in valid_names)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Overwrite the `__cat_dim__` method to handle sparse matrices to handle the names specified in `is_valid`.

        Parameters
        ----------
        key : str
            Key of the data.
        value : Any
            Value of the data.
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The concatenation dimension.
        """
        if torch_geometric.utils.is_sparse(value) and self.is_valid(key):
            return (0, 1)
        elif "index" in key or key == "face":
            return -1
        else:
            return 0




def collate_fn(batch):
    r"""Overwrite `torch_geometric.data.DataLoader` collate function to use the `DomainData` class.

    This ensures that the `torch_geometric` dataloaders work with sparse matrices that are not necessarily named `adj`. The function also generates the batch slices for the different cell dimensions.

    Parameters
    ----------
    batch : list
        List of data objects (e.g., `torch_geometric.data.Data`).

    Returns
    -------
    torch_geometric.data.Batch
        A `torch_geometric.data.Batch` object.
    """
    data_list = []
    batch_idx_dict = defaultdict(list)

    # Keep track of the running index for each cell dimension
    running_idx = {}

    for batch_idx, b in enumerate(batch):
        values, keys = b[0], b[1]
        data = DomainData()
        for key, value in zip(keys, values, strict=False):
            if torch_geometric.utils.is_sparse(value):
                value = value.coalesce()
            data[key] = value

        # Generate batch_slice values for x_1, x_2, x_3, ...
        x_keys = [el for el in keys if ("x_" in el)]
        for x_key in x_keys:
            if x_key != "x_0":
                if x_key != "x_hyperedges":
                    cell_dim = int(x_key.split("_")[1])
                else:
                    cell_dim = x_key.split("_")[1]

                current_number_of_cells = data[x_key].shape[0]

                batch_idx_dict[f"batch_{cell_dim}"].append(
                    torch.tensor([[batch_idx] * current_number_of_cells])
                )

                if (
                    running_idx.get(f"cell_running_idx_number_{cell_dim}")
                    is None
                ):
                    running_idx[f"cell_running_idx_number_{cell_dim}"] = (
                        current_number_of_cells
                    )

                else:
                    running_idx[f"cell_running_idx_number_{cell_dim}"] += (
                        current_number_of_cells
                    )

        data_list.append(data)

    batch = torch_geometric.data.Batch.from_data_list(data_list)

    # Rename batch.batch to batch.batch_0 for consistency
    batch["batch_0"] = batch.pop("batch")

    # Add batch slices to batch
    for key, value in batch_idx_dict.items():
        batch[key] = torch.cat(value, dim=1).squeeze(0).long()

    # Ensure shape is torch.Tensor
    # "shape" describes the number of n_cells in each graph
    if batch.get("shape") is not None:
        cell_statistics = batch.pop("shape")
        batch["cell_statistics"] = torch.Tensor(cell_statistics).long()

    return batch
