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



@hydra.main(version_base=None, config_path="conf", config_name="config")
def setup_training(config):
    pl.seed_everything(config.my_model.seed)

  # Instantiate and load dataset
    
    dataset_loader = GraphLoader(DictConfig(      
        {"data_dir": "../data",
          "data_name": config.dataset.loader.parameters.data_name, 
          "split_type": config.dataset.split_params.split_type}), 
            transforms = True)
    

    dataset, dataset_dir = dataset_loader.load()

    # Preprocess dataset and load the splits
    
    transform_config = None
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    train_data, validation_data, test_data = preprocessor.load_dataset_splits(config.dataset.split_params)

    datamodule = TBXDataloader(train_data, validation_data, test_data, batch_size=config.dataset.dataloader_params.batch_size)

    # loader = GraphLoader(
    #     {"data_dir": "../data",
    #       "data_name": config.dataset.loader.parameters.data_name, 
    #       "split_type": config.dataset.split_params.split_type},
    #         transforms = True)
    
    # datasets = loader.load()

    # train_data, validation_data, test_data = datasets

    # train_loader = DataLoader(
    #     train_data, 
    #     batch_size=config.dataset.dataloader_params.batch_size, 
    #     shuffle=config.dataset.dataloader_params.get('shuffle', True), 
    #     #collate_fn=custom_collate, 
    #     num_workers=config.dataset.dataloader_params.num_workers,
    #     pin_memory=config.dataset.dataloader_params.pin_memory
    #     )
    
    # val_loader = DataLoader(
    #     validation_data, 
    #     batch_size=config.dataset.dataloader_params.batch_size, 
    #     shuffle=False, 
    #     #collate_fn=custom_collate, 
    #     num_workers=config.dataset.dataloader_params.num_workers,
    #     pin_memory=config.dataset.dataloader_params.pin_memory
    #     )

    #wandb_logger = WandbLogger(project='experiments-with-hydra')
    hparams = create_hyperparameters(config)

    #wandb_logger.log_hyperparams(hparams)

    channel = Model_channel(hparams)
    trainer = pl.Trainer(max_epochs=config.training.max_epochs, accelerator = "cpu", )#logger=wandb_logger, log_every_n_steps=10)

    trainer.fit(channel, datamodule)
    #trainer = None
    
    return trainer, channel, datamodule



@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_and_plot(config: DictConfig):
    validation_accuracies = []
    validation_std_devs = []

    for ratio in config.exp.pooling_ratios:
        
        config.pooling.pooling_ratio = ratio  
        
        trainer, channel, datamodule = setup_training(config)
        trainer.fit(channel, datamodule)

        snr_accuracies = []
        snr_std_devs = []

        for snr in config.exp.test_snr_val:
            trial_accuracies = []

            for _ in range(config.exp.num_trials):
                channel.snr_db = snr
                test_result = trainer.validate(channel, datamodule) # ATTENZIONE: qui prima era val_loader, non datamodule.
                trial_accuracies.append(test_result[0]['val_acc'])

            average_accuracy = np.mean(trial_accuracies)
            std_dev_accuracy = np.std(trial_accuracies)

            snr_accuracies.append(average_accuracy)
            snr_std_devs.append(std_dev_accuracy) 
        
        validation_accuracies.append(snr_accuracies)
        validation_std_devs.append(snr_std_devs)

    plot_results(validation_accuracies, validation_std_devs, config.exp.test_snr_val, config.exp.pooling_ratios, config.pooling.pooling_type)




def inspect_model_output(channel, datamodule, batch_idx=0):
    """
    Passes one batch of data through the model and prints the output.

    Args:
    - channel (Model_channel): The untrained model instance.
    - datamodule (TBXDataloader): The datamodule containing the dataset splits.
    - batch_idx (int): Index of the batch to inspect.

    Returns:
    - None
    """

    # Set the model to evaluation mode (important if you have layers like dropout)
    channel.eval()

    # Get a single batch from the train dataloader
    dataloader = datamodule.train_dataloader()
    for i, batch in enumerate(dataloader):
        #if i == batch_idx:
            # Pass the batch through the model
        with torch.no_grad():  # Disable gradient computation
            output, edges, ne_probs = channel(batch)

        # Print the output
        print(f"Output from the model (batch {i}):")
        print(output)
        print("Output shape:", output.shape)
        print("Edges shape:", edges.shape if edges is not None else "None")
        print("Node probabilities (ne_probs):", ne_probs.shape if ne_probs is not None else "None")
        #break
    # else:
    #     print(f"Batch index {batch_idx} out of range for dataloader.")

    # Set the model back to training mode (if needed)
    channel.train()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    # Initialize the model, datamodule, etc.
    trainer, channel, datamodule = setup_training(config)

    # Inspect the model output for the first batch
    inspect_model_output(channel = channel, datamodule = datamodule)


if __name__ == "__main__":
    train_and_plot()
    #setup_training()
    # trainer, channel, train_loader, val_loader = setup_training()
    # trainer.fit(channel, train_dataloaders=train_loader, val_dataloaders=val_loader)


# if __name__ == "__main__":
#     main()