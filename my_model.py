import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch_geometric.nn import global_mean_pool
import pytorch_lightning as pl
from layers import MLP, GNN, DGM, NoiseBlock, DGM_d
from torch_geometric.nn.pool import TopKPooling, EdgePooling, SAGPooling, ASAPooling
import hydra
from omegaconf import DictConfig


class Model_channel(pl.LightningModule):
    def __init__(self, hparams):

        super(Model_channel, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        self.save_hyperparameters(hparams)
        self.pre = MLP(hparams["pre_layers"], dropout = hparams["dropout"])
        self.gnn = GNN(hparams["conv_layers"], dropout = hparams["dropout"])
        self.pooling_type = hparams["pooling"]
        if self.pooling_type == 'topk': 
            self.pool = TopKPooling(in_channels = hparams["conv_layers"][-1], ratio = hparams["ratio"],) #min_score = hparams["topk_minscore"])  #ratio arg will be ignored if min score is not none
        elif self.pooling_type == 'edge':
            self.pool = EdgePooling(in_channels = hparams["conv_layers"][-1])
        elif self.pooling_type == 'sag': 
            self.pool = SAGPooling(in_channels = hparams["conv_layers"][-1], ratio = hparams["ratio"])      
        elif self.pooling_type == 'asa': 
            self.pool = ASAPooling(in_channels = hparams["conv_layers"][-1], ratio = hparams["ratio"])      
        
          
        self.noise = NoiseBlock()
        #self.snr_db = hparams["snr_db"]
        self.snr_db = None    # in this way, a different snr value is sampled at every forward pass

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
            self.graph_f = DGM_d(
                GNN(hparams["dgm_layers"], dropout=hparams["dropout"]),
                k=hparams["k"],
                distance=hparams["distance"],
            )
        elif (not hparams["use_gcn"]) and (hparams["dgm_name"] == 'topk_dgm') :
            self.graph_f = DGM_d(
                MLP(hparams["dgm_layers"], dropout=hparams["dropout"]),
                k=hparams["k"],
                distance=hparams["distance"],

            )

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

    def forward(self, data):
        '''
        data: a batch of data. Must have attributes: x, batch, ptr
        '''

        
        x = data.x.detach()
        batch = data.batch_0
        ptr = data.ptr
        x = self.pre(x)
        # LTI
        if self.dgm_name == 'alpha_dgm':
            x_aux, edges, ne_probs = self.graph_f(x, data["edge_index"], batch, ptr)  #x, edges_hat, logprobs
        elif self.dgm_name == 'topk_dgm':
            x_aux, edges, ne_probs = self.graph_f(x, data["edge_index"]) 
        elif self.dgm_name == 'no_dgm': 
            # x_aux = self.graph_f(x, data["edge_index"]) 
            edges = data["edge_index"]
            ne_probs = None

        

        # FEATURE EXTRACTION
        x = self.gnn(x, edges)

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
        x = self.noise(x, self.snr_db)

        x = global_mean_pool(x, batch)  #aggregate all features in one supernode per graph.
        # TODO if noise is mean zero
        x = self.post(x)

        return x, edges, ne_probs

    def configure_optimizers(self):
        if self.hparams["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["lr"], momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer; choose 'adam' or 'sgd'.")


        return optimizer

# TODO: fix questi steps perchè le dimensioni non matchano. è come se il batch fosse sempre fatto da un solo datapoint, a prescindere dal batch size che gli do io 
# TODO: quando uso proteins si blocca tutto perchè aggiunge la quarta feature non so perché porca puttana

    # def training_step(self, batch, batch_idx):
    #     pred, _, _ = self(batch)
    #     train_lab = batch.y

    #     print(pred.shape)
    #     print(train_lab.shape)
    #     tr_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(train_lab).float())
    #     #tr_loss = F.binary_cross_entropy_with_logits(pred, train_lab)

    #     if torch.isnan(tr_loss).any() or torch.isnan(pred).any():
    #         print(f"NaN detected in training data or loss at batch {batch_idx}")
    #         print(f"Predictions: {pred}, Labels: {train_lab}")



    #     self.log("train_acc", self.train_acc(pred.softmax(-1).argmax(-1), train_lab), on_step=False, on_epoch=True, prog_bar = True)
    #     self.log("train_loss", tr_loss, on_step=False, on_epoch=True, prog_bar = True)
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    #     return tr_loss


    # def validation_step(self, batch, batch_idx):
    #     val_lab = batch.y
    #     pred, _, _ = self(batch)

        
    #     print(pred.shape)
    #     print(val_lab.shape)

    #     val_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(val_lab).float())
    #     #val_loss = F.binary_cross_entropy_with_logits(pred, val_lab)


    #     if torch.isnan(val_loss).any() or torch.isnan(pred).any():
    #         print(f"NaN detected in validation data or loss at batch {batch_idx}")
    #         print(f"Predictions: {pred}, Labels: {val_lab}")

    #     val_acc = self.val_acc(pred.softmax(-1).argmax(-1), val_lab)
    #     #self.validation_step_outputs.append(val_acc)

    #     self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar = True)
    #     self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar = True)

    # # def on_validation_epoch_end(self):
    # #   epoch_average = torch.stack(self.validation_step_outputs).mean()
    # #   self.log("validation_epoch_average", epoch_average)
    # #   self.validation_step_outputs.clear()  # free memory

    def training_step(self, batch, batch_idx):
       # print(f"Batch size: {batch.batch.max().item() + 1}")  # Number of graphs in the batch
        pred, _, _ = self(batch)
        train_lab = batch.y

   
        #pred = pred[batch.train_mask].float()

        tr_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(train_lab, num_classes = self.num_classes).float())
        #tr_loss = F.binary_cross_entropy_with_logits(pred, train_lab)

        if torch.isnan(tr_loss).any() or torch.isnan(pred).any():
            print(f"NaN detected in training data or loss at batch {batch_idx}")
            print(f"Predictions: {pred}, Labels: {train_lab}")

        self.log("train_acc", self.train_acc(pred.softmax(-1).argmax(-1), train_lab), on_step=False, on_epoch=True, prog_bar = True)
        self.log("train_loss", tr_loss, on_step=False, on_epoch=True, prog_bar = True)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return tr_loss

    def validation_step(self, batch, batch_idx):
        # Print batch size to check if multiple graphs are being batched

        pred, _, _ = self(batch)
        val_lab = batch.y

        #pred = pred[batch.val_mask].float()
        
        val_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(val_lab, num_classes = self.num_classes).float())
        
        # Check for NaN values in the loss or predictions
        if torch.isnan(val_loss).any() or torch.isnan(pred).any():
            print(f"NaN detected in validation data or loss at batch {batch_idx}")
            print(f"Predictions: {pred}, Labels: {val_lab}")

        # Compute and log validation accuracy
        val_acc = self.val_acc(pred.softmax(-1).argmax(-1), val_lab)
        
        # Log validation accuracy and loss
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return val_loss




    def test_step(self, batch, batch_idx):
        test_lab = batch.y
        pred, _, _ = self(batch)


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
