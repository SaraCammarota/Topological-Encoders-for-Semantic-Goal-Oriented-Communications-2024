from omegaconf import DictConfig, OmegaConf
import hydra
from my_model import Model_channel
import pytorch_lightning as pl
from layers import *
from utils import *
from torch.utils.data import DataLoader
from loaders import GraphLoader, PreProcessor, TBXDataloader
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
import matplotlib.pyplot as plt
from loaders import *
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint




@hydra.main(version_base=None, config_path="conf", config_name="config")
def setup_training(config):
    pl.seed_everything(config.my_model.seed)

    dataset_loader = GraphLoader(DictConfig(      
        {"data_dir": "./data",
          "data_name": config.dataset.loader.parameters.data_name, 
          "split_type": config.dataset.split_params.split_type}), 
            transforms = True)
    

    dataset, dataset_dir = dataset_loader.load()

    if config.dataset.loader.parameters.data_name == 'PROTEINS': 
        dataset_dir = '.'
    
    transform_config = None
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    train_data, validation_data, test_data = preprocessor.load_dataset_splits(config.dataset.split_params)

    datamodule = TBXDataloader(train_data, validation_data, test_data, batch_size=config.dataset.dataloader_params.batch_size)

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


    #wandb_logger = WandbLogger(project='experiments-with-hydra')
    hparams = create_hyperparameters(config)

    #wandb_logger.log_hyperparams(hparams)

    channel = Model_channel(hparams)
    trainer = pl.Trainer(max_epochs=config.training.max_epochs, accelerator = "cpu", callbacks=[early_stopping_callback, checkpoint_callback] )#logger=wandb_logger, log_every_n_steps=10)

    trainer.fit(channel, datamodule)

    best_model_path = checkpoint_callback.best_model_path
    best_model = Model_channel.load_from_checkpoint(best_model_path, hparams=hparams)


    return trainer, best_model, datamodule



@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_and_plot(config: DictConfig):
    validation_accuracies = []
    validation_std_devs = []

    for ratio in config.exp.pooling_ratios:
        
        config.pooling.pooling_ratio = ratio  
        
        trainer, channel, datamodule = setup_training(config)
        #trainer.fit(channel, datamodule)

        snr_accuracies = []
        snr_std_devs = []

        for snr in config.exp.test_snr_val:
            trial_accuracies = []

            for _ in range(config.exp.num_trials):
                channel.snr_db = snr
                test_result = trainer.validate(channel, datamodule) 
                trial_accuracies.append(test_result[0]['val_acc'])

            average_accuracy = np.mean(trial_accuracies)
            std_dev_accuracy = np.std(trial_accuracies)

            snr_accuracies.append(average_accuracy)
            snr_std_devs.append(std_dev_accuracy) 
        
        validation_accuracies.append(snr_accuracies)
        validation_std_devs.append(snr_std_devs)

    plot_results(validation_accuracies, validation_std_devs, config.exp.test_snr_val, config.exp.pooling_ratios, config.pooling.pooling_type, config.dataset.loader.parameters.data_name)







if __name__ == "__main__":
    

    train_and_plot()
    #setup_training()
