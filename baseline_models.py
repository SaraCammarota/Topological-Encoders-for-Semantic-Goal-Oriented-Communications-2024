import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch_geometric.nn import global_mean_pool
import pytorch_lightning as pl
from layers import MLP, NoiseBlock, KMeans, PCAReconstructor
from baseline_layers import * 
import hydra
from omegaconf import DictConfig
import warnings
from layers import MLP, GNN, DGM, NoiseBlock, DGM_d
from torch_geometric.nn.pool import TopKPooling, EdgePooling, SAGPooling, ASAPooling
from torch_geometric.nn import knn_graph





class MLP_KMeans(pl.LightningModule):
    def __init__(self, hparams):

        super(MLP_KMeans, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        self.save_hyperparameters(hparams)
        self.pre = MLP(hparams["pre_layers"], dropout = hparams["dropout"])   
    
        self.pooling_ratio = hparams["ratio"]

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
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     x, batch = self.cluster(x, batch)    # centroids are returned

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

        #tr_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(train_lab, num_classes = self.num_classes).float())
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

        pred = self(batch)
        val_lab = batch.y

        #val_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(val_lab, num_classes = self.num_classes).float())
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







class MLP_PCA(pl.LightningModule):
    def __init__(self, hparams):

        super(MLP_PCA, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        self.save_hyperparameters(hparams)
        self.pre = MLP(hparams["pre_layers"][:-1] + [hparams["num_features"]], dropout = hparams["dropout"])   
    
        self.pooling_ratio = hparams["ratio"]

        self.pca_dim = max(1, round(self.pooling_ratio * hparams["num_features"]))        
        self.pca = PCAReconstructor(self.pca_dim)

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
        self.receiver = MLP([self.pca_dim] + hparams['receiver_layers'][1:], dropout= hparams["dropout"])


    def forward(self, data):
        '''
        data: a batch of data. Must have attributes: x, batch, ptr
        '''

        x = data.x.detach()
        batch = data.batch_0
        ptr = data.ptr

        x = self.pre(x)

        x = self.pca(x)
        
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




class Perceiver_channel(pl.LightningModule):
    def __init__(self, hparams):

        super().__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        self.save_hyperparameters(hparams)
        self.num_classes = hparams["num_classes"]
        self.num_features = hparams["num_features"]
        self.pooling_ratio = hparams["ratio"]

        self.avg_num_nodes = hparams["avg_num_nodes"]

        compressed_dim = max(1, round(self.pooling_ratio * self.avg_num_nodes))  # --->  self.num_datapoints is different for each graph. 

        
        self.perceiver = Perceiver(input_channels=self.num_features,
                                   latent_dim = self.num_features)
    
        
        self.noisy_training = hparams["noisy_training"]
        self.noise = NoiseBlock()
        self.snr_db = None    # in this way, a different snr value is sampled at every forward pass
        
        #self.post = MLP(hparams['post_layers'], dropout= hparams["dropout"])
        self.post = nn.Linear(self.num_features, self.num_classes)
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
        

        self.receiver = MLP(hparams['receiver_layers'][1:], dropout= hparams["dropout"])
                                                #num_latents        #latents_dim
        self.latents = nn.Parameter(torch.randn(compressed_dim, self.num_features, ))



    def forward(self, data):
        '''
        data: a batch of data. Must have attributes: x, batch, ptr
        '''

        x = data.x.detach()
        batch = data.batch_0
        ptr = data.ptr

        b = max(batch) + 1
        latents = repeat(self.latents, "n d -> b n d", b=b)

        x = self.perceiver(x, latents, batch)

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

        #x = self.receiver(x)

        x = x.mean(-2)    #equivalent to global mean pooling in this case

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

        #tr_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(train_lab, num_classes = self.num_classes).float())
        tr_loss = F.cross_entropy(pred, train_lab)

        if torch.isnan(tr_loss).any() or torch.isnan(pred).any():
            print(f"NaN detected in training data or loss at batch {batch_idx}")
            print(f"Predictions: {pred}, Labels: {train_lab}")
        batch_size = batch.batch_0.max().item() + 1  
        self.log("train_acc", self.train_acc(pred.softmax(-1).argmax(-1), train_lab), on_step=False, on_epoch=True, prog_bar = True, batch_size = batch_size)
        self.log("train_loss", tr_loss, on_step=False, on_epoch=True, prog_bar = True, batch_size = batch_size)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return tr_loss

    def validation_step(self, batch, batch_idx):

        pred = self(batch)
        val_lab = batch.y

        # val_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(val_lab, num_classes = self.num_classes).float())
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





class MLP_Bottleneck(pl.LightningModule):
    def __init__(self, hparams):

        super(MLP_Bottleneck, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        self.save_hyperparameters(hparams)

        self.pooling_ratio = hparams["ratio"]
        bottleneck = max(1, round(self.pooling_ratio * hparams["num_features"]))
        self.pre = MLP(hparams["pre_layers"] + [bottleneck], dropout = hparams["dropout"])   


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
        self.receiver = MLP([bottleneck] + hparams['receiver_layers'], dropout= hparams["dropout"])


    def forward(self, data):
        '''
        data: a batch of data. Must have attributes: x, batch, ptr
        '''
        
        x = data.x.detach()
        batch = data.batch_0
        ptr = data.ptr

        x = self.pre(x)

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

        #tr_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(train_lab, num_classes = self.num_classes).float())
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

        pred = self(batch)
        val_lab = batch.y

        #val_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(val_lab, num_classes = self.num_classes).float())
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





# -----------



class Knn_channel(pl.LightningModule):
    def __init__(self, hparams):

        super(Knn_channel, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        self.save_hyperparameters(hparams)
        self.pre = MLP(hparams["pre_layers"], dropout = hparams["dropout"])
        self.gnn = GNN(hparams["conv_layers"][:-1] + [hparams["pre_layers"][0]], dropout = hparams["dropout"])
        self.pooling_type = hparams["pooling"]
        self.pooling_ratio = hparams["ratio"]
        if self.pooling_type == 'topk': 
            self.pool = TopKPooling(in_channels = hparams["pre_layers"][0], ratio = self.pooling_ratio) #min_score = hparams["topk_minscore"])  #ratio arg will be ignored if min score is not none
        elif self.pooling_type == 'edge':
            self.pool = EdgePooling(in_channels = hparams["pre_layers"][0])
        elif self.pooling_type == 'sag': 
            self.pool = SAGPooling(in_channels = hparams["pre_layers"][0], ratio = self.pooling_ratio)      
        elif self.pooling_type == 'asa': 
            self.pool = ASAPooling(in_channels = hparams["pre_layers"][0], ratio = self.pooling_ratio)      
        
        self.noisy_training = hparams["noisy_training"]
        self.noise = NoiseBlock()
        #self.snr_db = hparams["snr_db"]
        self.snr_db = None    # in this way, a different snr value is sampled at every forward pass

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

        self.receiver = GNN(hparams['receiver_layers'], dropout= hparams["dropout"])    # TODO MLP or GNN?

    def forward(self, data):
        '''
        data: a batch of data. Must have attributes: x, batch, ptr
        '''
        
        x = data.x.detach()
        batch = data.batch_0
        ptr = data.ptr
        x = self.pre(x)
        
        # LTI    done with knn instead of DGM

        edges = knn_graph(x, k = 5, batch = batch)
        print(edges)

        if self.skip is not None: 
            skip = self.skip(x)

        # FEATURE EXTRACTION
        x = self.gnn(x, edges)

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

        if self.noisy_training == True: 
            x = self.noise(x, self.snr_db)

        elif self.noisy_training == False:
            if self.training == False:
                x = self.noise(x, self.snr_db)

        else:
            print('Invalid self.noisy_training value')
            print(f"self.noisy_training: {self.noisy_training} (type: {type(self.noisy_training)})")


        x = self.receiver(x, edges)

        #x = torch.nn.functional.relu(x)

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
