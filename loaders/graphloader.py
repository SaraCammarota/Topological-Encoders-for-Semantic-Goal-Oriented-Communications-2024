
import torch
from torch.utils.data import Subset
import numpy as np

from abc import ABC
import os
from torch.utils.data import Dataset
import torch_geometric
from sklearn.model_selection import StratifiedKFold
import torch_geometric.transforms as T





class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AbstractLoader(ABC):
    def __init__(self):
        super().__init__()


class GraphLoader(AbstractLoader):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        # Still not instantiated

    def load(self):
        data_dir = os.path.join(
            self.parameters["data_dir"], self.parameters["data_name"]
        )



# could add QM9 dataset https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html

        if self.parameters["data_name"] in [
            "MUTAG",
            "ENZYMES",
            "PROTEINS",
            #"COLLAB",
            #"IMDB-BINARY",
            #"IMDB-MULTI",
            "REDDIT-BINARY",
            "NCI1",
            "NCI109",

        ]:


            dataset = torch_geometric.datasets.TUDataset(
                root=self.parameters["data_dir"],
                name=self.parameters["data_name"],
                use_node_attr=False,
                
            )


            dataset = load_graph_tudataset_split(dataset, self.parameters)

            return dataset

        elif self.parameters["data_name"] in ["CORA"]:

            dataset = torch_geometric.datasets.Planetoid(
                root=self.parameters["data_dir"],
                name=self.parameters["data_name"],
                
            )[0]

            split_idx = {"train": np.array(np.where(dataset.train_mask.bool()==1))[0]}
            split_idx["validation"] = np.array(np.where(dataset.val_mask.bool()==1))[0]
            split_idx["test"] = np.array(np.where(dataset.test_mask.bool()==1))[0]



        elif self.parameters["data_name"] in ["ZINC"]:
            datasets = []
            for split in ["train", "val", "test"]:
                datasets.append(
                    torch_geometric.datasets.ZINC(
                        root=self.parameters["data_dir"],
                        subset=True,
                        split=split,
                    )
                )

            assert self.parameters["split_type"] == "fixed"
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
            joined_dataset = datasets[0] + datasets[1] + datasets[2]

            # if self.transforms_config is not None:
            #     joined_dataset = Preprocessor(
            #         data_dir,
            #         joined_dataset,
            #         self.transforms_config,
            #     )

            # Split back the into train/val/test datasets
            dataset = get_train_val_test_graph_datasets(joined_dataset, split_idx)

        elif self.parameters["data_name"] in ["AQSOL"]:
            datasets = []
            for split in ["train", "val", "test"]:
                datasets.append(
                    torch_geometric.datasets.AQSOL(
                        root=self.parameters["data_dir"],
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
            joined_dataset = datasets[0] + datasets[1] + datasets[2]

            # if self.transforms_config is not None:
            #     joined_dataset = Preprocessor(
            #         data_dir,
            #         joined_dataset,
            #         self.transforms_config,
            #     )

            # Split back the into train/val/test datasets
            dataset = get_train_val_test_graph_datasets(joined_dataset, split_idx)
        else:
            raise NotImplementedError(
                f"Dataset {self.parameters['data_name']} not implemented"
            )

        return dataset

def k_fold_split(dataset, parameters, test_ratio = 0.2, ignore_negative=True):
    """
    Returns train and valid indices as in K-Fold Cross-Validation. If the split already exists it loads it automatically, otherwise it creates the split file for the subsequent runs.

    :param dataset: Dataset object containing either one or multiple graphs
    :param data_dir: The directory where the data is stored, it will be used to store the splits
    :param parameters: DictConfig containing the parameters for the dataset
    :param ignore_negative: If True the function ignores negative labels. Default True.
    :return split_idx: A dictionary containing "train" and "valid" tensors with the respective indices.
    """
    data_dir = parameters["data_split_dir"]
    k = parameters["k"]
    fold = parameters["data_seed"]
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
        if parameters["task_level"] == "graph":
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
