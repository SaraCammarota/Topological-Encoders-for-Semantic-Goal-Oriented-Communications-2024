import torch
import torch.nn as nn
from entmax import entmax15  
from omegaconf import DictConfig
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import os


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
    
    if config.dataset.loader.parameters.data_name == 'MUTAG': 
        num_features = config.dataset.parameters.num_features[0]
    else: 
        num_features = config.dataset.parameters.num_features
    
    num_classes = config.dataset.parameters.num_classes


    hyperparams = {
        "num_features": num_features,
        "num_classes": num_classes,
        "pre_layers" : [num_features] + [config.my_model.layers.get("hsize", 64) for _ in range(config.my_model.layers.get("n_pre", 2) )],
        "conv_layers": [config.my_model.layers.get("hsize", 64) for _ in range(config.my_model.layers.get("n_conv", 2) )],
        "post_layers" : [config.my_model.layers.get("hsize", 64) for _ in range(config.my_model.layers.get("n_post", 2) )] + [num_classes],
        "dgm_layers": [config.my_model.layers.get("hsize", 64) for _ in range(config.my_model.layers.get("n_dgm_layers", 2) + 1)],
        "lr": config.training.get('lr', 0.001),
        "dgm_name": config.dgm.get('name', 'no_dgm'),
        "gamma": config.dgm.get('gamma', 0),
        "std": config.dgm.get('std', 0),
        "k": config.dgm.get("k", 4),
        "distance": config.dgm.get('distance', 'euclidean'),
        "use_gcn": config.my_model.get('my_model.use_gcn', False),
        "dropout": config.my_model.get('dropout', 0.5),
        "ensemble_steps": config.my_model.get('my_model.ensemble_steps', 1),
        "optimizer": config.training.get('optimizer', 'adam'),
        "pooling": config.pooling.get('pooling_type', 'default_pooling'),
        "ratio": config.pooling.get('pooling_ratio', 0.5),
        "snr_db": config.my_model.channel.get('snr_db', 10),
        "skip_connection": config.my_model.get('skip_connection', False),
        "receiver_layers": [num_features] + [config.my_model.layers.get('hsize', 64) for _ in range(config.my_model.layers.get('receiver', 64) )],
        "noisy_training": config.training.get('noisy', False),
        "pca_dim": config.pooling.get('pca_dim', None),
    }

    return hyperparams






def custom_collate(batch):
    return Batch.from_data_list(batch)


def plot_results(validation_accuracies, validation_std_devs, snr_values, pooling_ratios, pooling_name, data_name, dgm_name):

    for idx, ratio in enumerate(pooling_ratios):
        plt.errorbar(snr_values, validation_accuracies[idx], yerr=validation_std_devs[idx], label=f'Ratio: {ratio}', capsize=5)

    plt.xlabel('Validate SNR (dB)')
    plt.ylabel('Validation/Best Accuracy')
    plt.title(f'Accuracy vs. SNR on {data_name}, with {pooling_name} pooling, no noise')
    plt.legend(title='Compression Levels')
    plt.savefig(f'new_plots/{data_name}/{pooling_name}/no_noise.png')
    plt.show()
    plt.close()


def plot_results_same(noisy_validation_accuracies, noisy_validation_std_devs, 
                      smooth_validation_accuracies, smooth_validation_std_devs, 
                      config):
    
    snr_values, pooling_ratios, pooling_name, data_name = config.exp.test_snr_val, config.exp.pooling_ratios, config.pooling.pooling_type, config.dataset.loader.parameters.data_name

    for idx, ratio in enumerate(pooling_ratios):
        color = f'C{idx}'
        
        plt.errorbar(snr_values, smooth_validation_accuracies[idx], yerr=smooth_validation_std_devs[idx], 
                     label=f'Pooling Ratio: {ratio}', color=color, capsize=5, linestyle='-', marker='o')

        plt.errorbar(snr_values, noisy_validation_accuracies[idx], yerr=noisy_validation_std_devs[idx], 
                     color=color, capsize=5, linestyle='--', marker='x')

    plt.xlabel('Validate SNR (dB)')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Accuracy vs. SNR on {data_name}, with {pooling_name} pooling')

    training_type_legend = [
        plt.Line2D([0], [0], color='black', linestyle='-', marker='o', label='Trained without Noise'),
        plt.Line2D([0], [0], color='black', linestyle='--', marker='x', label='Trained with Noise'),
    ]
    training_type_legend_plt = plt.legend(handles=training_type_legend, loc='best', title='Training Type')

    compression_legend = plt.legend(title="Pooling Ratio", loc='best')

    plt.gca().add_artist(training_type_legend_plt)
    
    plt.grid(True)

    folder_path = f'new_plots/use_gcn_{config.my_model.use_gcn}/{data_name}/{pooling_name}'
    os.makedirs(folder_path, exist_ok=True)

    plt.savefig(f'{folder_path}/prova.png')
    plt.show()
    plt.close()


