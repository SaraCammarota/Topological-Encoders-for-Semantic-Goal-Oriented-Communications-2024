import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from utils import *
import torch
#from pykeops.torch import LazyTensor   #I have problems with pykeops (no compiler?)


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




#Euclidean distance
def pairwise_euclidean_distances(x, dim=-1):
    dist = torch.cdist(x,x)**2
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
    def __init__(self, embed_f, k=5, distance="euclidean", sparse=True):
        super(DGM_d, self).__init__()
        
        self.sparse=sparse
        
        self.temperature = nn.Parameter(torch.tensor(1. if distance=="hyperbolic" else 4.).float())
        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.k = k
        self.distance = distance
        
        self.debug=False
        
    def forward(self, x, A, not_used=None, fixedges=None):
        if x.shape[0]==1:
            x = x[0]
        x = self.embed_f(x,A)
        if x.dim()==2:
            x = x[None,...]
    
        if self.training:
            if fixedges is not None:                
                return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)
            #sampling here
            edges_hat, logprobs = self.sample_without_replacement(x)
                
        else:
            with torch.no_grad():
                if fixedges is not None:                
                    return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)
                #sampling here
                edges_hat, logprobs = self.sample_without_replacement(x)
              
        if self.debug:
            if self.distance=="euclidean":
                D, _x = pairwise_euclidean_distances(x)
            if self.distance=="hyperbolic":
                D, _x = pairwise_poincare_distances(x)
                
            self.D = (D * torch.exp(torch.clamp(self.temperature,-5,5))).detach().cpu()
            self.edges_hat=edges_hat.detach().cpu()
            self.logprobs=logprobs.detach().cpu()
#             self.x=x

        return x, edges_hat, logprobs
    

    def sample_without_replacement(self, x):
        
        b,n,_ = x.shape
        
        if self.distance=="euclidean":
            G_i = LazyTensor(x[:, :, None, :])    # (M**2, 1, 2)
            X_j = LazyTensor(x[:, None, :, :])    # (1, N, 2)
        
            mD = ((G_i - X_j) ** 2).sum(-1)

            #argKmin already add gumbel noise
            lq = mD * torch.exp(torch.clamp(self.temperature,-5,5))
            indices = lq.argKmin(self.k, dim=1)

            x1 = torch.gather(x, -2, indices.view(indices.shape[0],-1)[...,None].repeat(1,1,x.shape[-1]))
            x2 = x[:,:,None,:].repeat(1,1,self.k,1).view(x.shape[0],-1,x.shape[-1])
            logprobs = (-(x1-x2).pow(2).sum(-1) * torch.exp(torch.clamp(self.temperature,-5,5))).reshape(x.shape[0],-1,self.k)

        if self.distance=="hyperbolic":
            pass
            x_norm = (x**2).sum(-1,keepdim=True)
            x_norm = (x_norm.sqrt()-1).relu() + 1 
            x = x/(x_norm*(1+1e-2)) #safe distance to the margin
            x_norm = (x**2).sum(-1,keepdim=True)
                
            G_i = LazyTensor(x[:, :, None, :])    # (M**2, 1, 2)
            X_j = LazyTensor(x[:, None, :, :])    # (1, N, 2)

            G_i2 = LazyTensor(1-x_norm[:, :, None, :])    # (M**2, 1, 2)
            X_j2 = LazyTensor(1-x_norm[:, None, :, :])    # (1, N, 2)

            pq = ((G_i - X_j) ** 2).sum(-1)
            N = (G_i2*X_j2)
            XX = (1e-6+1+2*pq/N)
            mD = (XX+(XX**2-1).sqrt()).log()**2

            lq = mD * torch.exp(torch.clamp(self.temperature,-5,5))
            indices = lq.argKmin(self.k, dim=1)

            x1 = torch.gather(x, -2, indices.view(indices.shape[0],-1)[...,None].repeat(1,1,x.shape[-1]))
            x2 = x[:,:,None,:].repeat(1,1,self.k,1).view(x.shape[0],-1,x.shape[-1])

            x1_n = torch.gather(x_norm, -2, indices.view(indices.shape[0],-1)[...,None].repeat(1,1,x_norm.shape[-1]))
            x2_n = x_norm[:,:,None,:].repeat(1,1,self.k,1).view(x.shape[0],-1,x_norm.shape[-1])

            pq = (x1-x2).pow(2).sum(-1)
            pqn = ((1-x1_n)*(1-x2_n)).sum(-1)
            XX = 1e-6+1+2*pq/pqn
            dist = torch.log(XX+(XX**2-1).sqrt())**2
            logprobs = (-dist * torch.exp(torch.clamp(self.temperature,-5,5))).reshape(x.shape[0],-1,self.k)

            if self.debug:
                self._x=x.detach().cpu()+0

        
        rows = torch.arange(n).view(1,n,1).to(x.device).repeat(b,1,self.k)
        edges = torch.stack((indices.view(b,-1),rows.view(b,-1)),-2)

        if self.sparse:
            return (edges+(torch.arange(b).to(x.device)*n)[:,None,None]).transpose(0,1).reshape(2,-1), logprobs
        return edges, logprobs
    