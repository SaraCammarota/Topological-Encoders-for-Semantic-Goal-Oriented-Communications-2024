import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from utils import *
import torch
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.decomposition import PCA as SKPCA
from torch_geometric.utils import dense_to_sparse

class GNN(nn.Module):
    def __init__(self, layers_size, dropout=0.0, last_act=False):
        super(GNN, self).__init__()
        self.d = dropout
        self.convs = nn.ModuleList()
        self.last_act = last_act
        for li in range(1, len(layers_size)):
            self.convs.append(GCNConv(layers_size[li - 1], layers_size[li]))
    def forward(self, x, e, weights=None):
        for i, c in enumerate(self.convs):
            x = c(F.dropout(x, p=self.d, training=self.training), e, weights)
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




# this has to change to DGMc and it will return also the edge weights
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
        
    def forward(self, x, A, batch = None, not_used=None, fixedges=None):
        x = self.embed_f(x,A)  

        if self.training:

            # if fixedges is not None:                
            #     return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)

            D, _x = self.distance(x)
            if batch is None:
            #sampling here
                edges_hat, logprobs = self.sample_without_replacement(D)
            elif batch is not None: 
                edges_hat, logprobs = self.sample_without_replacement_batch(D, batch)
                
        else:
            with torch.no_grad():
                # if fixedges is not None:                
                #     return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)
                D, _x = self.distance(x)

                #sampling here

                if batch is None:
                #sampling here
                    edges_hat, logprobs = self.sample_without_replacement(D)
                elif batch is not None: 
                    edges_hat, logprobs = self.sample_without_replacement_batch(D, batch)

              
        if self.debug:
            self.D = D
            self.edges_hat=edges_hat
            self.logprobs=logprobs
            self.x=x

        return x, edges_hat, logprobs
    

    def sample_without_replacement(self, logits):

        # logits is a matrix tot_num_nodes_batch x tot_num_nodes_batch with pairwise distances between nodes. 

        b,n,_ = logits.shape
#         logits = logits * torch.exp(self.temperature*10)
        logits = logits * torch.exp(torch.clamp(self.temperature,-5,5))
        
        q = torch.rand_like(logits) + 1e-8
        lq = (logits-torch.log(-torch.log(q)))
        logprobs, indices = torch.topk(-lq,self.k)  
    
        rows = torch.arange(n).view(1,n,1).to(logits.device).repeat(b,1,self.k)
        edges = torch.stack((indices.view(b,-1),rows.view(b,-1)),-2)
        
        if self.sparse:
            return (edges+(torch.arange(b).to(logits.device)*n)[:,None,None]).transpose(0,1).reshape(2,-1), logprobs
        return edges, logprobs


    def sample_without_replacement_batch(self, logits, batch):
        """
        This function samples k edges without replacement for each graph in the batch.
        Args:
            logits: Tensor of shape (total_nodes, total_nodes) containing the logits for each pair of nodes.
            batch: Tensor of shape (total_nodes,) indicating which graph each node belongs to.
        Returns:
            edges: Tensor of shape (2, total_edges) containing the sampled edges.
            logprobs: Tensor of shape (batch_size, k) containing the log probabilities of the sampled edges.
        """

        device = logits.device
        unique_graphs = batch.unique(sorted=True)
        edges_list = []
        logprobs_list = []

        logits = logits * torch.exp(torch.clamp(self.temperature, -5, 5))
        q = torch.rand_like(logits) + 1e-8 
        lq = logits - torch.log(-torch.log(q))
        


        for graph_id in unique_graphs:


            mask = (batch == graph_id)
            lq_i = lq[mask][:, mask] 
            num_nodes_i = lq_i.size(0) 
            k = self.k
            if num_nodes_i < k: 
                print(f"Overwriting TopKDGM Configuration. Graph {graph_id} has {num_nodes_i} nodes and selected value for k was {k}.")


                logprobs, indices = torch.topk(lq_i, num_nodes_i, largest = False)  # i topk edge per il grafo i-esimo (largest = False per selezionare i più piccoli 
                                                                         # ovvero quelli più simili)

            else: 
                logprobs, indices = torch.topk(lq_i, self.k, largest = False)

            rows = torch.arange(num_nodes_i, device=device).view(-1, 1).expand_as(indices)
            
            edges = torch.stack((rows.reshape(-1), indices.view(-1))) 
        
            global_indices = mask.nonzero(as_tuple=True)[0]
            
            edges = global_indices[edges]
            
            edges_list.append(edges)
            logprobs_list.append(logprobs)
        
        first_elements = torch.cat([edge[0] for edge in edges_list], dim=0)
        second_elements = torch.cat([edge[1] for edge in edges_list], dim=0)

        all_edges = torch.stack((first_elements, second_elements), dim=0)
        
        #all_logprobs = torch.cat(logprobs_list, dim=0)
        all_logprobs = None
        #there is a problem with cat when self.k gets modified

        # if self.sparse:
        #     all_edges = all_edges.transpose(0, 1).reshape(2, -1)
        
        return all_edges, all_logprobs




class DGM_c(nn.Module):
    
    input_dim = 32

    def __init__(self, embed_f, k=None, distance="euclidean"):
        super(DGM_c, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(1).float())
        self.threshold = nn.Parameter(torch.tensor(0.5).float())
        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.distance = distance
        
        self.scale = nn.Parameter(torch.tensor(-1).float(),requires_grad=False)
        self.centroid = nn.Parameter(torch.zeros((1,DGM_c.input_dim)).float(),requires_grad=True)
        
        
    def forward(self, x, A, not_used=None, fixedges=None):
        
        x = self.embed_f(x,A)  
        
        # estimate normalization parameters
        if self.scale <0:            
            self.centroid.data = x.mean(-2,keepdim=True).detach()
            self.scale.data = (0.9/(x-self.centroid).abs().max()).detach()
            #print(f"Centroid shape: {self.centroid.shape}")

        x_centered = x - self.centroid
        #print(f"x_centered shape: {x_centered.shape}")
        
        if self.distance=="hyperbolic":
            D, _x = pairwise_poincare_distances((x-self.centroid)*self.scale)
        else:
            D, _x = pairwise_euclidean_distances(x_centered*self.scale)
            
        A = torch.sigmoid(self.temperature*(self.threshold.abs()-D))

        edge_index, edge_weight = dense_to_sparse(A)
        

#         self.A=A
#         A = A/A.sum(-1,keepdim=True)
        return x, edge_index, edge_weight



class DGM_c_batch(nn.Module):
    
    def __init__(self, dgm_c: DGM_c):
        super(DGM_c_batch, self).__init__()
        self.dgm_c = dgm_c  # DGM_c for one graph

    def forward(self, x, edge_index, batch):
        """
        Forward pass for batched graphs.
        x: Node features [num_nodes, feature_dim]
        edge_index: Edge index [2, num_edges]
        batch: Batch tensor that indicates which graph each node belongs to
        """
        num_nodes = x.size(0)  # Total number of nodes
        num_graphs = batch.max().item() + 1  # Number of graphs in the batch
        
        # Split the batch into individual graphs
        x_list = []
        edge_index_list = []
        edge_weight_list = []
        node_offset = 0
        
        for graph_id in range(num_graphs):
            # Select the nodes and edges that belong to the current graph
            graph_mask = (batch == graph_id)
            x_graph = x[graph_mask]  # Features of nodes in this graph
            
            # Get the edges for the current graph
            edge_mask = (batch[edge_index[0]] == graph_id) & (batch[edge_index[1]] == graph_id)
            edge_index_graph = edge_index[:, edge_mask] - node_offset  # Adjust edge indices for local nodes

            # Apply DGM_c for the current graph
            x_graph, edge_index_graph, edge_weight_graph = self.dgm_c(x_graph, edge_index_graph)

            # Adjust edge indices to the global batch space
            edge_index_graph = edge_index_graph + node_offset
            
            # Append results
            x_list.append(x_graph)
            edge_index_list.append(edge_index_graph)
            edge_weight_list.append(edge_weight_graph)

            # Update node offset for the next graph
            node_offset += graph_mask.sum().item()

        # Concatenate results from all graphs
        x_batch = torch.cat(x_list, dim=0)
        edge_index_batch = torch.cat(edge_index_list, dim=1)
        edge_weight_batch = torch.cat(edge_weight_list, dim=0)

        return x_batch, edge_index_batch, edge_weight_batch


class KMeans(nn.Module):
    def __init__(self, ratio, init='k-means++', max_iter=300, tol=1e-4, n_init=10, pca_dim=None):
        
        super(KMeans, self).__init__()
        self.ratio = ratio
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.pca_dim = pca_dim  # Number of components for PCA, if None, PCA is skipped
        self.kmeans = None

    def forward(self, X, batch):
        """
        Perform K-means clustering on the input data X for each graph and adjust batch information.

        Args:
            X (torch.Tensor): Input data of shape (N, D), where N is the number of samples and D is the feature dimension.
            batch (torch.Tensor): Batch tensor indicating the graph each node belongs to.

        Returns:
            centroids (torch.Tensor): Cluster centers of shape (n_clusters_total, D).
            new_batch (torch.Tensor): Updated batch tensor reflecting the new batch assignment for centroids.
        """
        unique_graphs = batch.unique()
        centroids_list = []
        new_batch_list = []

        with torch.no_grad():
            for graph_id in unique_graphs:
                graph_mask = batch == graph_id
                X_graph = X[graph_mask]

                X_graph_np = X_graph.cpu().numpy()

                if self.pca_dim is not None and X_graph_np.shape[1] > self.pca_dim:
                    pca = SKPCA(n_components=self.pca_dim)
                    X_graph_np = pca.fit_transform(X_graph_np)

                n_clusters = max(1, round(self.ratio * X_graph_np.shape[0]))  

                self.kmeans = SklearnKMeans(n_clusters=n_clusters, init=self.init, max_iter=self.max_iter, tol=self.tol, n_init=self.n_init)
                self.kmeans.fit(X_graph_np)

                centroids_np = self.kmeans.cluster_centers_
                centroids = torch.tensor(centroids_np, dtype=X.dtype, device=X.device)

                centroids_list.append(centroids)

                new_batch_list.append(torch.full((centroids.shape[0],), graph_id, dtype=batch.dtype, device=batch.device))

        centroids = torch.cat(centroids_list, dim=0)
        new_batch = torch.cat(new_batch_list, dim=0)

        return centroids, new_batch
    


class PCAReconstructor:
    def __init__(self, q=5, niter=2):
        self.q = q
        self.niter = niter
        
        self.mean_ = None
    
    def decompose(self, A):
        mean = A.mean(dim=1, keepdim=True)            
        U, S, V = torch.pca_lowrank(A, q=self.q, center=True, niter=self.niter)
        return U, S, V, mean

    def reconstruct(self, U, S, V, mean):    
        SV = torch.einsum('bq,bdq->bqd', S, V)
        A_reconstructed = U @ SV

        if self.center and mean is not None:
            A_reconstructed += mean
        return A_reconstructed

    def compute_reconstruction_loss(self, A):
        U, S, V, mean = self.decompose(A)    
        A_reconstructed = self.reconstruct(U, S, V, mean)
        loss = ((A - A_reconstructed) ** 2).sum()
        return loss
    
    def project(self, A):
        """Decompose A and project it to the principal components"""
        torch.random.manual_seed(42)
        U, S, V, A_mean = self.decompose(A)
        A_projected = torch.einsum('bnd,bdi->bni', (A - A_mean), V)
        return A_projected, V, A_mean
    
    def reconstruct_from_projection(self, A_projected, V, A_mean):
        """Reconstruct A from the projected data"""
        A_reconstructed = torch.einsum('bni,bdi->bnd', A_projected, V) + A_mean
        return A_reconstructed
    

