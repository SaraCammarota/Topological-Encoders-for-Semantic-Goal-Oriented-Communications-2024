import torch
import torch.nn as nn
from entmax import entmax15  
from omegaconf import DictConfig
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
    


class LayerNorm(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = nn.Parameter(gamma * torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        if x.size(-1) == 1:
            std = 1
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def entmax(x: torch.tensor, ln, std=0):

    probs = -torch.cdist(x, x)   # compute 2-norm distances
    probs = probs + torch.empty(probs.size()).normal_(mean=0, std=std)
    # use alpha entmax after layer normalization ln

    normalized = ln(probs)

    vprobs = entmax15(normalized.fill_diagonal_(-1e-6), dim=-1)

    # make the graph simmetric
    res = (((vprobs + vprobs.t()) / 2) > 0) * 1
    # t_() is an in place operator that transposes the tensor
    edges = res.nonzero().t_()
    logprobs = res.sum(dim=1)
    return edges, logprobs


def entmax_batch(x: torch.Tensor, batch: torch.Tensor, ptr, ln, std=0):

    '''
    Computes graph from auxiliary features
    x: auxiliary features of all graphs in the batch
    batch: the list of dimension total_n_nodes specifying to which graph of the batch each node belongs to
    ptr: list containing the number of nodes of each graph in batch
    ln: instance of LayerNorm
    std: standard deviation of normal

    '''

    edge_list = []
    logprob_list = []
    #num_graphs = batch.max().item() + 1
    num_graphs = len(ptr) - 1
    start_index = ptr

    for i in range(num_graphs):

        mask = (batch == i)
        nodes_features = x[mask]
        probs = -torch.cdist(nodes_features, nodes_features)
        probs = probs + torch.empty(probs.size(), device=x.device).normal_(mean=0, std=std)
        vprobs = entmax15(ln(probs).fill_diagonal_(-1e-6), dim=-1)
        res = (((vprobs + vprobs.t()) / 2) > 0) * 1

        edges = res.nonzero(as_tuple=False).t_()
        edges += start_index[i]
        logprobs = res.sum(dim=1)

        edge_list.append(edges)
        logprob_list.append(logprobs)

    edges = torch.cat(edge_list, dim=1)
    logprobs = torch.cat(logprob_list)

    return edges, logprobs



def create_hyperparameters(config: DictConfig):
    num_features = config.dataset.parameters.num_features
    num_classes = config.dataset.parameters.num_classes

    hyperparams = {
        "num_features": num_features,
        "num_classes": num_classes,
        "pre_layers" : [num_features] + [config.my_model.layers.hsize for _ in range(config.my_model.layers.n_pre)],
        "conv_layers": [config.my_model.layers.hsize for _ in range(config.my_model.layers.n_conv)],
        "post_layers" : [config.my_model.layers.hsize for _ in range(config.my_model.layers.n_post)] + [num_classes],
        "dgm_layers": [config.my_model.layers.hsize for _ in range(config.my_model.layers.n_dgm_layers + 1)],
        "lr": config.training.lr,
        #"k": 4,  
        "dgm_name": config.dgm.name, 
        "gamma": config.dgm.get("gamma", 0),
        "std": config.dgm.get("std", 0),
        "k": config.dgm.get("k", 0),
        "distance": config.get("dgm.distance", 'euclidean'),
        "use_gcn": config.my_model.use_gcn,
        "dropout": config.my_model.dropout,
        "ensemble_steps": config.my_model.ensemble_steps,
        "optimizer": "adam",
        "pooling": config.pooling.pooling_type,
        "ratio": config.pooling.pooling_ratio,
        "topk_minscore": config.pooling.topk_minscore,
        "snr_db": config.my_model.channel.snr_db
    }


    return hyperparams





def custom_collate(batch):
    return Batch.from_data_list(batch)


def plot_results(validation_accuracies, validation_std_devs, snr_values, pooling_ratios, pooling_name):

    for idx, ratio in enumerate(pooling_ratios):
        plt.errorbar(snr_values, validation_accuracies[idx], yerr=validation_std_devs[idx], label=f'Ratio: {ratio}', capsize=5)

    plt.xlabel('Validate SNR (dB)')
    plt.ylabel('Validation/Best Accuracy')
    plt.title('Accuracy vs. SNR')
    plt.legend(title='Compression Levels')
    plt.show()
    plt.savefig(f'accuracy_vs_snr_{pooling_name}.png')
    plt.close()




class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

                