from omegaconf import DictConfig, OmegaConf
import hydra
from my_model import Model_channel
import pytorch_lightning as pl
from layers import *
from utils import *
from torch.utils.data import DataLoader
from loaders import GraphLoader, PreProcessor
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
import matplotlib.pyplot as plt
from loaders import *



@hydra.main(version_base=None, config_path="conf", config_name="config")
def setup_training(config):
    pl.seed_everything(config.my_model.seed)

  # Instantiate and load dataset
    
    dataset_loader = GraphLoader(DictObj(
        {"data_dir": "../data",
          "data_name": config.dataset.loader.parameters.data_name, 
          "split_type": config.dataset.split_params.split_type}), 
            transforms = True)
    

    dataset, dataset_dir = dataset_loader.load()
    # Preprocess dataset and load the splits
    
    transform_config = None
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    train_data, validation_data, test_data = preprocessor.load_dataset_splits(config.dataset.split_params)

    # loader = GraphLoader(
    #     {"data_dir": "../data",
    #       "data_name": config.dataset.loader.parameters.data_name, 
    #       "split_type": config.dataset.split_params.split_type},
    #         transforms = True)
    
    # datasets = loader.load()

    # train_data, validation_data, test_data = datasets

    train_loader = DataLoader(
        train_data, 
        batch_size=config.dataset.dataloader_params.batch_size, 
        shuffle=config.dataset.dataloader_params.get('shuffle', True), 
        #collate_fn=custom_collate, 
        num_workers=config.dataset.dataloader_params.num_workers,
        pin_memory=config.dataset.dataloader_params.pin_memory
        )
    
    val_loader = DataLoader(
        validation_data, 
        batch_size=config.dataset.dataloader_params.batch_size, 
        shuffle=False, 
        #collate_fn=custom_collate, 
        num_workers=config.dataset.dataloader_params.num_workers,
        pin_memory=config.dataset.dataloader_params.pin_memory
        )

    wandb_logger = WandbLogger(project='experiments-with-hydra')
    hparams = create_hyperparameters(config)
    wandb_logger.log_hyperparams(hparams)

    channel = Model_channel(hparams)
    trainer = pl.Trainer(max_epochs=config.training.max_epochs, logger=wandb_logger, log_every_n_steps=10)
    
    return trainer, channel, train_loader, val_loader



@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_and_plot(config: DictConfig):
    validation_accuracies = []
    validation_std_devs = []

    for ratio in config.exp.pooling_ratios:
        
        config.pooling.pooling_ratio = ratio  
        
        trainer, channel, train_loader, val_loader = setup_training(config)
        trainer.fit(channel, train_dataloaders=train_loader, val_dataloaders=val_loader)

        snr_accuracies = []
        snr_std_devs = []

        for snr in config.exp.test_snr_val:
            trial_accuracies = []

            for _ in range(config.exp.num_trials):
                channel.snr_db = snr
                test_result = trainer.test(channel, val_loader)
                trial_accuracies.append(test_result[0]['test_acc'])

            average_accuracy = np.mean(trial_accuracies)
            std_dev_accuracy = np.std(trial_accuracies)

            snr_accuracies.append(average_accuracy)
            snr_std_devs.append(std_dev_accuracy) 
        
        validation_accuracies.append(snr_accuracies)
        validation_std_devs.append(snr_std_devs)

    plot_results(validation_accuracies, validation_std_devs, config.exp.test_snr_val, config.exp.pooling_ratios, config.pooling.pooling_type)





if __name__ == "__main__":
    train_and_plot()
    # trainer, channel, train_loader, val_loader = setup_training()
    # trainer.fit(channel, train_dataloaders=train_loader, val_dataloaders=val_loader)


