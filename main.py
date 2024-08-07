from omegaconf import DictConfig, OmegaConf
import hydra
from my_model import Model_channel
import pytorch_lightning as pl
from layers import *
from utils import *
from torch.utils.data import DataLoader
from loaders import GraphLoader
    


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    
    pl.seed_everything(config.my_model.seed)

    loader = GraphLoader(
        {"data_dir": "../data",
          "data_name": config.dataset.loader.parameters.data_name, 
          "split_type": config.dataset.split_params.split_type},
            transforms = True)
    
    datasets = loader.load()

    train_data, validation_data, test_data = datasets

    train_loader = DataLoader(
        train_data, 
        batch_size=config.dataset.dataloader_params.batch_size, 
        shuffle=config.dataset.dataloader_params.get('shuffle', True), 
        collate_fn=custom_collate, 
        num_workers=config.dataset.dataloader_params.num_workers,
        pin_memory=config.dataset.dataloader_params.pin_memory
        )
    
    val_loader = DataLoader(
        validation_data, 
        batch_size=config.dataset.dataloader_params.batch_size, 
        shuffle=False, 
        collate_fn=custom_collate, 
        num_workers=config.dataset.dataloader_params.num_workers,
        pin_memory=config.dataset.dataloader_params.pin_memory
        )
    
    test_loader = DataLoader(
        test_data, 
        batch_size=config.dataset.dataloader_params.batch_size, 
        shuffle=False, 
        collate_fn=custom_collate, 
        num_workers=config.dataset.dataloader_params.num_workers,
        pin_memory=config.dataset.dataloader_params.pin_memory
        )

    
    hparams = create_hyperparameters(config)
    channel = Model_channel(hparams)
    
    trainer = pl.Trainer(max_epochs=config.training.max_epochs)  
    trainer.fit(channel, train_dataloaders=train_loader, val_dataloaders=val_loader)



#TODO: make different config files for different poolings and integrate in the hparams and self.pool of the model
#TODO: create functions for repeating the experiment and plotting the accuracy.


if __name__ == "__main__":
    main()

