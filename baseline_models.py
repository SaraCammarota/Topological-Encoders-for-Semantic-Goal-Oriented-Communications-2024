import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch_geometric.nn import global_mean_pool
import pytorch_lightning as pl
from layers import MLP, NoiseBlock, KMeans
import hydra
from omegaconf import DictConfig




class MLP_KMeans(pl.LightningModule):
    def __init__(self, hparams):

        super(MLP_KMeans, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        self.save_hyperparameters(hparams)
        self.pre = MLP(hparams["pre_layers"], dropout = hparams["dropout"])     # this is the same 
    
        self.pooling_ratio = hparams["ratio"]
              
        # TODO QUI IN QUALCHE MODO IL CLUSTERING

        self.cluster = KMeans(self.pooling_ratio) 
        
        self.noisy_training = hparams["noisy_training"]
        self.noise = NoiseBlock()
        self.snr_db = None    # in this way, a different snr value is sampled at every forward pass
        
        self.post = MLP(hparams['post_layers'], dropout= hparams["dropout"])

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
        self.receiver = MLP(hparams['receiver_layers'], dropout= hparams["dropout"])


    def forward(self, data):
        '''
        data: a batch of data. Must have attributes: x, batch, ptr
        '''
        
        x = data.x.detach()
        batch = data.batch_0
        ptr = data.ptr

        x = self.pre(x)

        # CLUSTERING    instead of pooling
        x, batch = self.cluster(x, batch)    # centroids are returned


        #AWG (ADDITIVE WHITE GAUSSIAN) NOISE

        if self.noisy_training == True: 
            x = self.noise(x, self.snr_db)

        elif self.noisy_training == False:
            if self.training == False:
                x = self.noise(x, self.snr_db)

        else:
            print('Invalid self.noisy_training value')
            print(f"self.noisy_training: {self.noisy_training} (type: {type(self.noisy_training)})")


        # RECEIVER
        x = self.receiver(x)
        x = global_mean_pool(x, batch)
        x = self.post(x)           

        return x


    def configure_optimizers(self):
        if self.hparams["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        elif self.hparams["optimizer"] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["lr"], momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer; choose 'adam' or 'sgd'.")


        return optimizer



    def training_step(self, batch, batch_idx):
       
        pred = self(batch)
        train_lab = batch.y

        tr_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(train_lab, num_classes = self.num_classes).float())

        if torch.isnan(tr_loss).any() or torch.isnan(pred).any():
            print(f"NaN detected in training data or loss at batch {batch_idx}")
            print(f"Predictions: {pred}, Labels: {train_lab}")
        batch_size = batch.batch_0.max().item() + 1  
        self.log("train_acc", self.train_acc(pred.softmax(-1).argmax(-1), train_lab), on_step=False, on_epoch=True, prog_bar = True, batch_size = batch_size)
        self.log("train_loss", tr_loss, on_step=False, on_epoch=True, prog_bar = True, batch_size = batch_size)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return tr_loss

    def validation_step(self, batch, batch_idx):

        pred = self(batch)
        val_lab = batch.y

        val_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(val_lab, num_classes = self.num_classes).float())
        
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
        pred = self(batch)


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
