import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch_geometric.nn import global_mean_pool
import pytorch_lightning as pl
from layers import MLP, GNN, DGM, NoiseBlock, DGM_d, DGM_c, DGM_c_batch, DGM_Lev_batch, DGM_Lev
from torch_geometric.nn.pool import TopKPooling, EdgePooling, SAGPooling, ASAPooling
import hydra
from omegaconf import DictConfig
from torchmetrics.audio import SignalNoiseRatio




class Model_channel(pl.LightningModule):
    def __init__(self, hparams):

        super(Model_channel, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        self.save_hyperparameters(hparams)
        self.pre = MLP(hparams["pre_layers"], dropout = hparams["dropout"])
        self.gnn = GNN(hparams["conv_layers"][:-1] + [hparams["pre_layers"][0]], dropout = hparams["dropout"])
        self.pooling_type = hparams["pooling"]
        self.pooling_ratio = hparams["ratio"]
        if self.pooling_type == 'topk': 
            self.pool = TopKPooling(in_channels = hparams["pre_layers"][0], ratio = float(self.pooling_ratio)) #min_score = hparams["topk_minscore"])  #ratio arg will be ignored if min score is not none
        elif self.pooling_type == 'edge':
            self.pool = EdgePooling(in_channels = hparams["pre_layers"][0])
        elif self.pooling_type == 'sag': 
            self.pool = SAGPooling(in_channels = hparams["pre_layers"][0], ratio = float(self.pooling_ratio))      
        elif self.pooling_type == 'asa': 
            self.pool = ASAPooling(in_channels = hparams["pre_layers"][0], ratio = float(self.pooling_ratio))     
        else:
            print(f'{self.pooling_type} not implemented.') 
        
        self.noisy_training = hparams["noisy_training"]
        self.noise = NoiseBlock()
        self.snr_db = hparams["snr_db"]


        if hparams["use_gcn"] and (hparams["dgm_name"] == 'alpha_dgm'):
            self.graph_f = DGM(
                GNN(hparams["dgm_layers"], dropout=hparams["dropout"]),
                gamma=hparams["gamma"],
                std=hparams["std"],
            )
        elif (not hparams["use_gcn"]) and (hparams["dgm_name"] == 'alpha_dgm') :
            self.graph_f = DGM(
                MLP(hparams["dgm_layers"], dropout=hparams["dropout"]),
                gamma=hparams["gamma"],
                std=hparams["std"],

            )

        elif hparams["use_gcn"] and (hparams["dgm_name"] == 'topk_dgm'):
            self.graph_f = DGM_c_batch(DGM_c(
                GNN(hparams["dgm_layers"], dropout=hparams["dropout"]),
                k=hparams["k"],
                distance=hparams["distance"],
            ))
        elif (not hparams["use_gcn"]) and (hparams["dgm_name"] == 'topk_dgm') :
            self.graph_f = DGM_c_batch(DGM_c(
                MLP(hparams["dgm_layers"], dropout=hparams["dropout"]),
                k=hparams["k"],
                distance=hparams["distance"],

            ))

        elif hparams["use_gcn"] and (hparams["dgm_name"] == 'new_dgm'):
            print('Im using the GCN here')
            self.graph_f = DGM_Lev_batch(DGM_Lev(
                GNN(hparams["dgm_layers"], dropout=hparams["dropout"]),
                k=hparams["k"],
                distance=hparams["distance"],
            ))
        elif (not hparams["use_gcn"]) and (hparams["dgm_name"] == 'new_dgm') :
            print('Im not using the GCN here')
            self.graph_f = DGM_Lev_batch(DGM_Lev(
                MLP(hparams["dgm_layers"], dropout=hparams["dropout"]),
                k=hparams["k"],
                distance=hparams["distance"],

            ))

        # elif hparams["dgm_name"] == 'no_dgm':
        #     self.graph_f = GNN(hparams["dgm_layers"], dropout=hparams["dropout"])

        self.dgm_name = hparams["dgm_name"]

        self.post = MLP(hparams["post_layers"], dropout = hparams["dropout"])   

        self.avg_accuracy = None

        if hparams["num_classes"] > 2:
            self.train_acc = Accuracy(task='multiclass', num_classes=hparams["num_classes"], average='macro')
            self.val_acc = Accuracy(task='multiclass', num_classes=hparams["num_classes"], average='macro')
            self.test_acc = Accuracy(task='multiclass', num_classes=hparams["num_classes"], average='macro')
        elif hparams["num_classes"] == 2:
            self.train_acc = Accuracy(task='binary', num_classes=2)
            self.val_acc = Accuracy(task='binary', num_classes=2)
            self.test_acc = Accuracy(task='binary', num_classes=2)
        else:

            raise ValueError(f"Invalid number of classes ({hparams['num_classes']}).")
        
        self.num_classes = hparams["num_classes"]

        if hparams["skip_connection"] == True: 
            self.skip = torch.nn.Linear(hparams["pre_layers"][-1], hparams["pre_layers"][0])
        elif hparams["skip_connection"] == False: 
            self.skip = None

        self.receiver = MLP(hparams['receiver_layers'], dropout= hparams["dropout"])   

    def forward(self, data):
        '''
        data: a batch of data. Must have attributes: x, batch, ptr
        '''
        x = data.x.detach()
        batch = data.batch_0
        # ptr = data.ptr
        x = self.pre(x)

        if self.skip is not None: 
            skip = self.skip(x)

        # LTI

        if self.dgm_name == 'alpha_dgm':
            x_aux, edges, ne_probs = self.graph_f(x, data.edge_index, batch)  #x, edges_hat, logprobs
            x = self.gnn(x, edges)

        elif self.dgm_name == 'topk_dgm':

            x_aux, edges, edge_weights = self.graph_f(x, data.edge_index, batch) 
            x = self.gnn(x, edges, edge_weights)
            
        elif self.dgm_name == 'no_dgm': 
            # x_aux = self.graph_f(x, data.edge_index) 
            edges = data.edge_index
            ne_probs = None
            x = self.gnn(x, edges)

        elif self.dgm_name == 'new_dgm': 

            x_aux, edges, edge_weights = self.graph_f(x, data.edge_index, batch) 
            # x_aux, edges, edge_weights = self.graph_f(x_aux, edges, batch) 
            x = self.gnn(x, edges, edge_weights)

        # # FEATURE EXTRACTION -- moved above
        # x = self.gnn(x, edges, edge_weights)

        # Here one might add a skip connection 
        if self.skip is not None: 
            x = x + skip

        # POOLING  -- how is compression rate defined in this case? rho = k/n where k: num nodes in input; n: num nodes in output. num_features is fixed for now.
        # compression is the "ratio" argument.

        pool_output = self.pool(x = x, edge_index = edges, batch = batch)
        x = pool_output[0]
        edges = pool_output[1]
        if self.pooling_type in ['topk', 'sag', 'asa']: 
            batch = pool_output[3]
        elif self.pooling_type == 'edge':
            batch = pool_output[2]

        #AWG (ADDITIVE WHITE GAUSSIAN) NOISE
        # what if we don't take into account the noise during training?

        # print(f'before noise {self.snr_db}')
        # print(x)
        if self.noisy_training == True: 
            x = self.noise(x, batch, self.snr_db)

        elif self.noisy_training == False:
            if self.training == False:
                x = self.noise(x, batch, self.snr_db)
                # snrmod = SignalNoiseRatio()
                # print(f'this is the true snr {snrmod(x_n, x)}')
                # x = x_n


        else:
            print('Invalid self.noisy_training value')
            print(f"self.noisy_training: {self.noisy_training} (type: {type(self.noisy_training)})")

        # print(f'after noise {self.snr_db}')
        # print(x)
        x = self.receiver(x, edges)


        #x = torch.nn.functional.relu(x)

        x = global_mean_pool(x, batch)  #aggregate all features in one supernode per graph.
        x = self.post(x, edges)

        return x, edges#, ne_probs

    def configure_optimizers(self):
        if self.hparams["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"], weight_decay=1e-3)
        elif self.hparams["optimizer"] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["lr"], momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer; choose 'adam' or 'sgd'.")


        return optimizer



    def training_step(self, batch, batch_idx):
       
        pred, _ = self(batch)
        train_lab = batch.y


        tr_loss = F.cross_entropy(pred, train_lab)

        if torch.isnan(tr_loss).any() or torch.isnan(pred).any():
            print(f"NaN detected in training data or loss at batch {batch_idx}")
            # print(f"Predictions: {pred}, Labels: {train_lab}")
        batch_size = batch.batch_0.max().item() + 1  
        self.log("train_acc", self.train_acc(pred.softmax(-1).argmax(-1), train_lab), on_step=False, on_epoch=True, prog_bar = True, batch_size = batch_size)
        self.log("train_loss", tr_loss, on_step=False, on_epoch=True, prog_bar = True, batch_size = batch_size)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return tr_loss

    def validation_step(self, batch, batch_idx):

        pred, _ = self(batch)
        val_lab = batch.y

        #pred = pred[batch.val_mask].float()
        
        val_loss = F.cross_entropy(pred, val_lab)
        
        # Check for NaN values in the loss or predictions
        if torch.isnan(val_loss).any() or torch.isnan(pred).any():
            print(f"NaN detected in validation data or loss at batch {batch_idx}")
            # print(f"Predictions: {pred}, Labels: {val_lab}")
        batch_size = batch.batch_0.max().item() + 1  
        # Compute and log validation accuracy
        val_acc = self.val_acc(pred.softmax(-1).argmax(-1), val_lab)
        
        # Log validation accuracy and loss
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size = batch_size)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size = batch_size)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return val_loss




    def test_step(self, batch, batch_idx):
        test_lab = batch.y
        pred, _ = self(batch)


        for _ in range(1, self.hparams.ensemble_steps):
            pred_, _, _ = self(batch)
            pred += pred_

        self.test_acc(pred.softmax(-1).argmax(-1), test_lab)
        self.log(
            "test_acc",
            self.test_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=test_lab.size(0),
        )
