import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch_geometric.nn import global_mean_pool
import pytorch_lightning as pl
from layers import MLP
from baseline_layers import * 
from layers import MLP, GNN, DGM
from torch_geometric.nn import knn_graph
from layers import MLP, GNN, DGM, DGM_c, DGM_c_batch
import hydra
from omegaconf import DictConfig
import hydra
from my_model import Model_channel
import pytorch_lightning as pl
from layers import *
from utils import *
from loaders import GraphLoader, PreProcessor, TBXDataloader
from loaders import *
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from baseline_models import Perceiver_channel, MLP_Bottleneck, Knn_channel



class simple_knn(pl.LightningModule):
    def __init__(self, hparams):

        super(simple_knn, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        self.save_hyperparameters(hparams)
        self.pre = MLP(hparams["pre_layers"], dropout = hparams["dropout"])
        self.gnn = GNN(hparams["conv_layers"], dropout = hparams["dropout"])

        self.k = hparams['num_edges']

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

        self.post = MLP(hparams['post_layers'], dropout = hparams["dropout"])

    def forward(self, data):
        '''
        data: a batch of data. Must have attributes: x, batch, ptr
        '''
        
        x = data.x.detach()
        batch = data.batch_0
        ptr = data.ptr
        x = self.pre(x)
        
        # LTI    done with knn instead of DGM

        edges = knn_graph(x, k = self.k, batch = batch)

        if self.skip is not None: 
            skip = self.skip(x)

        # FEATURE EXTRACTION
        x = self.gnn(x, edges)

        # Here one might add a skip connection 
        if self.skip is not None: 
            x = x + skip

        x = global_mean_pool(x, batch)  #aggregate all features in one supernode per graph.
        
        x = self.post(x, edges)

        return x, edges

    def configure_optimizers(self):
        if self.hparams["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["lr"], momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer; choose 'adam' or 'sgd'.")


        return optimizer



    def training_step(self, batch, batch_idx):
       
        pred, _  = self(batch)
        train_lab = batch.y

        tr_loss = F.cross_entropy(pred, train_lab)

        if torch.isnan(tr_loss).any() or torch.isnan(pred).any():
            print(f"NaN detected in training data or loss at batch {batch_idx}")
            print(f"Predictions: {pred}, Labels: {train_lab}")
        batch_size = batch.batch_0.max().item() + 1  
        self.log("train_acc", self.train_acc(pred.softmax(-1).argmax(-1), train_lab), on_step=False, on_epoch=True, prog_bar = True, batch_size = batch_size)
        self.log("train_loss", tr_loss, on_step=False, on_epoch=True, prog_bar = True, batch_size = batch_size)
        #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return tr_loss

    def validation_step(self, batch, batch_idx):

        pred, _ = self(batch)
        val_lab = batch.y

        #pred = pred[batch.val_mask].float()
        
        val_loss = F.cross_entropy(pred, val_lab)
        
        # Check for NaN values in the loss or predictions
        if torch.isnan(val_loss).any() or torch.isnan(pred).any():
            print(f"NaN detected in validation data or loss at batch {batch_idx}")
            print(f"Predictions: {pred}, Labels: {val_lab}")
        batch_size = batch.batch_0.max().item() + 1  
        # Compute and log validation accuracy
        val_acc = self.val_acc(pred.softmax(-1).argmax(-1), val_lab)
        
        # Log validation accuracy and loss
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size = batch_size)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size = batch_size)

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





class simple_dgm(pl.LightningModule):
    def __init__(self, hparams):

        super(simple_dgm, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        self.save_hyperparameters(hparams)
        self.pre = MLP(hparams["pre_layers"], dropout = hparams["dropout"])
        self.gnn = GNN(hparams["conv_layers"], dropout = hparams["dropout"])
        

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


    def forward(self, data):
        '''
        data: a batch of data. Must have attributes: x, batch, ptr
        '''
        x = data.x.detach()
        batch = data.batch_0
        ptr = data.ptr
        x = self.pre(x)

        if self.skip is not None: 
            skip = self.skip(x)

        # LTI

        if self.dgm_name == 'alpha_dgm':
            x_aux, edges, ne_probs = self.graph_f(x, data["edge_index"], batch, ptr)  #x, edges_hat, logprobs
            x = self.gnn(x, edges)

        elif self.dgm_name == 'topk_dgm':

            x_aux, edges, edge_weights = self.graph_f(x, data["edge_index"], batch) 
            x = self.gnn(x, edges, edge_weights)
            
        elif self.dgm_name == 'no_dgm': 
            # x_aux = self.graph_f(x, data["edge_index"]) 
            edges = data["edge_index"]
            ne_probs = None
            x = self.gnn(x, edges)


        # # FEATURE EXTRACTION -- moved above
        # x = self.gnn(x, edges, edge_weights)

        # Here one might add a skip connection 
        if self.skip is not None: 
            x = x + skip

  

        #x = torch.nn.functional.relu(x)

        x = global_mean_pool(x, batch)  #aggregate all features in one supernode per graph.
        x = self.post(x, edges)

        return x, edges#, ne_probs

    def configure_optimizers(self):
        if self.hparams["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
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


@hydra.main(version_base=None, config_path="conf", config_name="knn_dgm_config")

# we can do a "test" with IMDB-BINARY where the avg number of nodes per graph is 19.8 --> if we compress 50%, latent dimension is 10.

def setup_training(config):
    pl.seed_everything(config.my_model.seed)

    dataset_loader = GraphLoader(DictConfig(      
        {"data_dir": "./data",
          "data_name": config.dataset.loader.parameters.data_name, 
          "split_type": config.dataset.split_params.split_type, 
          "use_node_attributes": config.dataset.loader.parameters.get("use_node_attributes", False)}), 
            transforms = True)
    

    dataset, dataset_dir = dataset_loader.load()

    if config.dataset.loader.parameters.data_name == 'PROTEINS': 
        dataset_dir = '.'
    
    transform_config = None
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    train_data, validation_data, test_data = preprocessor.load_dataset_splits(config.dataset.split_params)

    datamodule = TBXDataloader(train_data, validation_data, test_data, batch_size=config.dataset.dataloader_params.batch_size,
                               num_workers=config.dataset.dataloader_params.num_workers)

    early_stopping_callback = EarlyStopping(
        monitor=config.training.early_stopping.monitor,
        patience=config.training.early_stopping.patience,
        mode=config.training.early_stopping.mode,
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=config.training.early_stopping.monitor, 
        filename='{epoch:02d}-{val_loss:.2f}',  
        save_top_k=1,  
        mode=config.training.early_stopping.mode, 
        save_weights_only=True  
    )


    #wandb_logger = WandbLogger(project='imdb_new_dgm')
    # wandb_logger = WandbLogger(project='mine_vs_baseline_PROTEINS_nodgm')
    hparams = create_hyperparameters(config)
    #wandb_logger.log_hyperparams(hparams)

    if config.my_model.name == 'simple_dgm':
        model = simple_dgm(hparams)
        trainer = pl.Trainer(max_epochs=config.training.max_epochs, accelerator = "cpu", callbacks=[checkpoint_callback, early_stopping_callback])#, logger=wandb_logger, log_every_n_steps=2)
        trainer.fit(model, datamodule)
        best_model_path = checkpoint_callback.best_model_path
        best_model = simple_dgm.load_from_checkpoint(best_model_path, hparams=hparams)

    if config.my_model.name == 'simple_knn':
        model = simple_knn(hparams)
        trainer = pl.Trainer(max_epochs=config.training.max_epochs, accelerator = "cpu", callbacks=[checkpoint_callback, early_stopping_callback])#, logger=wandb_logger, log_every_n_steps=2)
        trainer.fit(model, datamodule)
        best_model_path = checkpoint_callback.best_model_path
        best_model = simple_knn.load_from_checkpoint(best_model_path, hparams=hparams)


    else:
        print('Model not implemented. Check config file')


    #channel = MLP_PCA(hparams)
    # best_model = Model_channel.load_from_checkpoint(best_model_path, hparams=hparams)
    #best_model = MLP_PCA.load_from_checkpoint(best_model_path, hparams=hparams)
    
    return trainer, best_model, datamodule




if __name__ == "__main__":

    setup_training()

