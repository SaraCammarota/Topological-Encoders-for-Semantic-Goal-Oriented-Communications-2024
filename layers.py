import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from utils import *
import torch

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




# #Euclidean distance
# def pairwise_euclidean_distances(x, dim=-1):
#     dist = torch.cdist(x,x)**2
#     return dist, x

def pairwise_euclidean_distances(x, batch):
    # Batch is a vector indicating the graph each node belongs to
    dist = torch.zeros(x.size(0), x.size(0)).to(x.device)
    for graph_id in batch.unique():
        mask = (batch == graph_id)
        dist[mask][:, mask] = torch.cdist(x[mask], x[mask])**2
    return dist, x

# #Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x**2).sum(dim,keepdim=True)
    x_norm = (x_norm.sqrt()-1).relu() + 1 
    x = x/(x_norm*(1+1e-2))
    x_norm = (x**2).sum(dim,keepdim=True)
    
    pq = torch.cdist(x,x)**2
    dist = torch.arccosh(1e-6+1+2*pq/((1-x_norm)*(1-x_norm.transpose(-1,-2))))**2
    return dist, x

def sparse_eye(size):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    values = torch.tensor(1.0).float().expand(size)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size])) 


class DGM_d(nn.Module):
    def __init__(self, embed_f, k=5, distance=pairwise_euclidean_distances, sparse=True):
        super(DGM_d, self).__init__()
        
        self.sparse=sparse
        
        self.temperature = nn.Parameter(torch.tensor(1. if distance=="hyperbolic" else 4.).float())
        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.k = k
        
        self.debug=False
        if distance == 'euclidean':
            self.distance = pairwise_euclidean_distances
        else:
            self.distance = pairwise_poincare_distances
        
    def forward(self, x, A, not_used=None, fixedges=None):
        x = self.embed_f(x,A)  
        print('questo è il valore di x')
        print(x)
        print(x.shape)
        if self.training:
            if fixedges is not None:                
                return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)
            
            D, _x = self.distance(x)
           
            #sampling here
            edges_hat, logprobs = self.sample_without_replacement(D)
                
        else:
            with torch.no_grad():
                if fixedges is not None:                
                    return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)
                D, _x = self.distance(x)

                #sampling here

                print('questo è il valore di D distanza')
                print(D)
                print(D.shape)
                edges_hat, logprobs = self.sample_without_replacement(D)

              
        if self.debug:
            self.D = D
            self.edges_hat=edges_hat
            self.logprobs=logprobs
            self.x=x

        return x, edges_hat, logprobs
    

#     def sample_without_replacement(self, logits):
#         # if logits.dim() == 2:
#         # # questo l'ho aggiunto io perche non funzionava niente. devo capire se c'e' un problema con il caricamento dei dati per cui non viene fuori la batch dimension
#         #     logits = logits.unsqueeze(0)
#         b,n,_ = logits.shape
# #         logits = logits * torch.exp(self.temperature*10)
#         logits = logits * torch.exp(torch.clamp(self.temperature,-5,5))
        
#         q = torch.rand_like(logits) + 1e-8
#         lq = (logits-torch.log(-torch.log(q)))
#         logprobs, indices = torch.topk(-lq,self.k)  
    
#         rows = torch.arange(n).view(1,n,1).to(logits.device).repeat(b,1,self.k)
#         edges = torch.stack((indices.view(b,-1),rows.view(b,-1)),-2)
        
#         if self.sparse:
#             return (edges+(torch.arange(b).to(logits.device)*n)[:,None,None]).transpose(0,1).reshape(2,-1), logprobs
#         return edges, logprobs
    
    def sample_without_replacement(self, logits, batch):
        b = batch.max().item() + 1
        n = logits.size(1)
        logits = logits * torch.exp(torch.clamp(self.temperature, -5, 5))

        # Mask out logits between nodes of different graphs
        for graph_id in batch.unique():
            mask = (batch == graph_id)
            logits[:, ~mask] = float('-inf')  # Mask out irrelevant nodes

        q = torch.rand_like(logits) + 1e-8
        lq = (logits - torch.log(-torch.log(q)))
        logprobs, indices = torch.topk(-lq, self.k)

        rows = torch.arange(n).view(1, n, 1).to(logits.device).repeat(b, 1, self.k)
        edges = torch.stack((indices.view(b, -1), rows.view(b, -1)), -2)

        if self.sparse:
            return (edges + (torch.arange(b).to(logits.device) * n)[:, None, None]).transpose(0, 1).reshape(2, -1), logprobs
        return edges, logprobs