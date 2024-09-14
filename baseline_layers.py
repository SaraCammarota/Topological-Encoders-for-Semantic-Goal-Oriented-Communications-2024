import torch
from torch import nn, einsum

from einops import rearrange, repeat

from torch_geometric.utils import to_dense_batch
import torch_geometric.nn as pygnn
from torch_geometric.nn import global_mean_pool

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.ReLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout), # why dropout after linear?
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0,):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        # In the contex variable we pass agg. node features
        h = self.heads

        q = self.to_q(x)
        
        context = default(context, x)

        # From the agg.node features obtain keys and values
        k, v = self.to_kv(context).chunk(2, dim=-1)
       
        # "b n (h d) -> (b h) n d" <--> process each head 
        # in the batch independently.
        # n - stand for numebr of nodes
        dim = len(x.shape)
        if dim == 3:
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        
        elif dim == 2:
            q, k, v = map(lambda t: rearrange(t, "b (h d) -> (b h) d", h=h), (q, k, v))
            sim = einsum("i d, j d -> i j", q, k) * self.scale


        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            # ???
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        
        if dim == 3:
            out = einsum("b i j, b j d -> b i d", attn, v)
            out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        
        elif dim == 2:
            out = einsum("i d, d j -> i j", attn, v)
            out = rearrange(out, "(b h) d -> b (h d)", h=h)

        return self.to_out(out)

#####

class latentAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None,
                 heads=8, dim_head=64,
                 dropout=0.0, ff_dropout=0.0, layer_norm=True):
        super().__init__()
        self.latent_attn = Attention(
            query_dim=query_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
           
            )

        self.latent_ff = FeedForward(query_dim, dropout=ff_dropout)
        self.layer_norm = layer_norm

        
        self.norm_attn = nn.LayerNorm(query_dim)
        self.norm_ff = nn.LayerNorm(query_dim)
            
    def forward(self, latents):
        h_attn = self.latent_attn( self.norm_attn(latents) ) + latents
        h_attn = self.latent_ff( self.norm_ff(h_attn) ) + latents
        # if self.layer_norm:
        #     h_attn = self.norm_attn(h_attn)
        return h_attn

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None,
                 heads=8, dim_head=64,
                 dropout=0.0, ff_dropout=0.0, layer_norm=True):
        super().__init__()
        self.latent_attn = Attention(
            query_dim=query_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
          
            )

        self.latent_ff = FeedForward(query_dim, dropout=ff_dropout)
        self.layer_norm = layer_norm

       
        self.norm_attn = nn.LayerNorm(query_dim)
        self.norm_ff = nn.LayerNorm(query_dim)
        self.input_context_norm = nn.LayerNorm(context_dim)
        
            
    def forward(self, latents, context, mask):
       
        h_attn = (self.latent_attn( self.norm_attn(latents),
            context=self.input_context_norm(context),
            mask=mask) + latents
            )
        h_attn = self.latent_ff( self.norm_ff(h_attn) ) + latents
        # if self.layer_norm:
        #     h_attn = self.norm_attn(h_attn)
        return h_attn
        
class Perceiver(nn.Module):
    def __init__(
        self,
        input_channels=3,
        latent_dim=512,
        cross_heads=1,
        latent_heads=4,
        cross_dim_head=2,
        latent_dim_head=2,
        attn_dropout=0.0,
        ff_dropout=0.0,
        perceiver_depth=1,
    ):

        super().__init__()

        input_dim = input_channels

        self.cross_attn = CrossAttention(
            query_dim=latent_dim,
            context_dim=input_dim,
            heads=cross_heads,
            dim_head=cross_dim_head,
            dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

        self.cross_attn_shared = CrossAttention(
            query_dim=latent_dim,
            context_dim=input_dim,
            heads=cross_heads,
            dim_head=cross_dim_head,
            dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        
        self.attent_modules = []
         
        for _ in range(perceiver_depth):
            self_attend = latentAttention(
            query_dim=latent_dim,
            heads=latent_heads,
            dim_head=latent_dim_head,
            dropout=attn_dropout,
            )
            
            self.attent_modules.append(self_attend)
            #self.attent_modules.append(self.cross_attn_shared)

        self.attent_modules = nn.ModuleList(self.attent_modules)


    def forward(self, h_local, latents, batch_batch):
        #x_dense, _ = to_dense_batch(x, batch_batch)
        # -------- Cross attention ---------
        h_dense, mask = to_dense_batch(h_local, batch_batch)    # ---> (num_graphs_in_batch, max_num_nodes, num_features)
        #h_dense = torch.concat([h_dense, x_dense], dim=-1)
        # First cross attention
        h_attn = self.cross_attn(latents=latents, context=h_dense, mask=mask)  
    
        
        # -------- Latent attention + shared cross attention ---------
        for idx, att in enumerate(self.attent_modules):
            h_attn = att(h_attn)
            # if (idx % 2) == 0:
            #     # latent attention
            #     h_attn = att(h_attn)
            # else:
            #     # shared cross attention (every second) layer
            #     h_attn = att(latents=h_attn, context=h_dense, mask=mask)
        
        return h_attn

