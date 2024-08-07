import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from utils import *
import torch
from torch import nn

class GNN(nn.Module):
    def __init__(self, layers_size, dropout=0.0, last_act=False):
        super(GNN, self).__init__()
        self.d = dropout
        self.convs = nn.ModuleList()
        self.last_act = last_act
        for li in range(1, len(layers_size)):
            self.convs.append(GCNConv(layers_size[li - 1], layers_size[li]))
    def forward(self, x, e):
        for i, c in enumerate(self.convs):
            x = c(F.dropout(x, p=self.d, training=self.training), e)
            if i == len(self.convs) and self.last_act is False:
                break
            x = x.relu()
        return x


class MLP(nn.Module):
    def __init__(self, layers_size, dropout=0.0, final_activation=False):
        super(MLP, self).__init__()
        layers = []
        for li in range(1, len(layers_size)):
            layers.append(nn.Dropout(p=dropout))
            linear_layer = nn.Linear(layers_size[li - 1], layers_size[li])
            #layers.append(nn.Linear(layers_size[li - 1], layers_size[li]))
            init.xavier_normal_(linear_layer.weight)
            layers.append(linear_layer)
            if li == len(layers_size) - 1 and not final_activation:
                continue
            layers.append(nn.ReLU())
        self.MLP = nn.Sequential(*layers)

    def forward(self, x, e=None):
        x = self.MLP(x)
        return x
    
class DGM(nn.Module):
    def __init__(self, embed_f: nn.Module, gamma, std):
        super(DGM, self).__init__()
        self.ln = LayerNorm(gamma)
        self.std = std
        self.embed_f = embed_f

    def forward(self, x, edges, batch, ptr):

        """
        x: data
        edges: edges list of the batch
        batch: the list of dimension total_n_nodes specifying to which graph of the batch each node belongs to
        ptr: list containing the number of nodes of each graph in batch

         """
        #x = self.embed_f(x) # compute auxiliary features with an MLP or GNN depending on the input embed_f
        #x = self.embed_f(x, edges) # compute auxiliary features with an MLP or GNN depending on the input embed_f
        x_aux = self.embed_f(x, edges)
        if batch is not None:
            edges_hat, logprobs = entmax_batch(x=x_aux, batch = batch, ptr = ptr, ln=self.ln, std=self.std)
        elif batch is None:
            edges_hat, logprobs = entmax(x=x_aux, ln = self.ln, std = self.std)

        return x_aux, edges_hat, logprobs
    



class NoiseBlock(nn.Module):
    def __init__(self,):
      super().__init__()

    @torch.no_grad()
    def generate_noise(self, x: torch.Tensor, snr_db=None):
      """
      Adds noise to the input according to the given signal to noise ratio.
      The input is assumed to be of shape (batch_size, sequence_length, hidden_dim).
      """
      #assert snr_linear > 0, "SNR must be greater than 0"
      # Compute the signal power
      signal_power = torch.mean(x ** 2, dim=-1, keepdim=True)

      # Compute the noise power
      # Convert SNR from dB to linear scale
      snr_linear = 10**(snr_db / 10)

      noise_power = signal_power / snr_linear.to(signal_power.device)

      # Compute the standard deviation of the noise
      std = torch.sqrt(noise_power)
      noise = torch.randn_like(x, requires_grad=False) * std

      return noise

    #@torch.no_grad()
    def forward(self, x: torch.Tensor, snr_db = 0):
      """
      Adds noise to the input according to the given signal to noise ratio.
      The input is assumed to be of shape (batch_size, sequence_length, hidden_dim).
      """
      if snr_db is None:
        # Sample snr from uniform distribution between 1, 10
        # This a lirear snr, to map it to db use the following SNR_linear=10^(SNR_db/10)
        snr_db = torch.randint(-10, 10, (1,))
      else:
        snr_db = torch.tensor(snr_db).unsqueeze(0)

      noise = self.generate_noise(x, snr_db=snr_db)

      # Add the noise to the input
      x = x + noise
      return x


