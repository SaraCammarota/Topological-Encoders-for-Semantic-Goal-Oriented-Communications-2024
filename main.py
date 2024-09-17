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
from baseline_models import MLP_KMeans, MLP_PCA, Perceiver_channel, MLP_Bottleneck


@hydra.main(version_base=None, config_path="conf", config_name="config")

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


    wandb_logger = WandbLogger(project='mutag_perceiver')
    # wandb_logger = WandbLogger(project='mine_vs_baseline_PROTEINS_nodgm')
    hparams = create_hyperparameters(config)
    wandb_logger.log_hyperparams(hparams)

    if config.my_model.name == 'dgm_channel':
        channel = Model_channel(hparams)
        trainer = pl.Trainer(max_epochs=config.training.max_epochs, accelerator = "cpu", callbacks=[checkpoint_callback], logger=wandb_logger, log_every_n_steps=2)
        trainer.fit(channel, datamodule)
        best_model_path = checkpoint_callback.best_model_path
        best_model = Model_channel.load_from_checkpoint(best_model_path, hparams=hparams)

    elif config.my_model.name == 'perceiver': 
        channel = Perceiver_channel(hparams)
        trainer = pl.Trainer(max_epochs=config.training.max_epochs, accelerator = "cpu", callbacks=[checkpoint_callback], logger=wandb_logger, log_every_n_steps=2)
        trainer.fit(channel, datamodule)
        best_model_path = checkpoint_callback.best_model_path
        best_model = Perceiver_channel.load_from_checkpoint(best_model_path, hparams=hparams) 

    elif config.my_model.name == 'mlp_bottleneck': 
        channel = MLP_Bottleneck(hparams)
        trainer = pl.Trainer(max_epochs=config.training.max_epochs, accelerator = "cpu", callbacks=[checkpoint_callback], logger=wandb_logger, log_every_n_steps=2)
        trainer.fit(channel, datamodule)
        best_model_path = checkpoint_callback.best_model_path
        best_model = MLP_Bottleneck.load_from_checkpoint(best_model_path, hparams=hparams) 

    # elif config.my_model.name == 'Knn_channel': 
    #     channel = Knn_channel(hparams)
    #     trainer = pl.Trainer(max_epochs=config.training.max_epochs, accelerator = "cpu", callbacks=[checkpoint_callback], logger=wandb_logger, log_every_n_steps=2)
    #     trainer.fit(channel, datamodule)
    #     best_model_path = checkpoint_callback.best_model_path
    #     best_model = Knn_channel.load_from_checkpoint(best_model_path, hparams=hparams) 


    else:
        print('Model not implemented. Check config file')


    #channel = MLP_PCA(hparams)
    # best_model = Model_channel.load_from_checkpoint(best_model_path, hparams=hparams)
    #best_model = MLP_PCA.load_from_checkpoint(best_model_path, hparams=hparams)
    
    return trainer, best_model, datamodule



@hydra.main(version_base=None, config_path="conf", config_name="baseline_config")
def train_and_plot(config: DictConfig):
    validation_accuracies = []
    validation_std_devs = []

    for ratio in config.exp.pooling_ratios:
        
        config.pooling.pooling_ratio = ratio  
        
        trainer, channel, datamodule = setup_training(config)

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

    plot_results(validation_accuracies, validation_std_devs, config.exp.test_snr_val, 
                 config.exp.pooling_ratios, config.pooling.pooling_type, config.dataset.loader.parameters.data_name, config.dgm.name)



def return_train_and_plot(config: DictConfig):
    validation_accuracies = []
    validation_std_devs = []

    for ratio in config.exp.pooling_ratios:
        
        config.pooling.pooling_ratio = ratio  
        
        trainer, channel, datamodule = setup_training(config)

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

    return validation_accuracies, validation_std_devs



@hydra.main(version_base=None, config_path="conf", config_name="baseline_config")
def train_and_plot_same(config: DictConfig):

    config.training.noisy = True
    

    noisy_validation_accuracies, noisy_validation_std_devs = return_train_and_plot(config)

    config.training.noisy = False

    smooth_validation_accuracies, smooth_validation_std_devs = return_train_and_plot(config)

    plot_results_same(noisy_validation_accuracies, noisy_validation_std_devs, smooth_validation_accuracies, smooth_validation_std_devs, config)


from matplotlib.lines import Line2D

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_and_plot_comparison(config: DictConfig):
    # Initialize lists to store validation accuracies and standard deviations for both methods
    perceiver_accuracies = []
    perceiver_std_devs = []
    my_model_accuracies = []
    my_model_std_devs = []

    # Loop over pooling ratios
    for ratio in config.exp.pooling_ratios:
        config.pooling.pooling_ratio = ratio

        # Test Perceiver model
        config.my_model.name = 'mlp_bottleneck'
        trainer_perceiver, perceiver_model, datamodule = setup_training(config)

        perceiver_snr_accuracies = []
        perceiver_snr_std_devs = []

        for snr in config.exp.test_snr_val:
            trial_accuracies = []

            for _ in range(config.exp.num_trials):
                perceiver_model.snr_db = snr
                test_result = trainer_perceiver.validate(perceiver_model, datamodule)
                trial_accuracies.append(test_result[0]['val_acc'])

            perceiver_snr_accuracies.append(np.mean(trial_accuracies))
            perceiver_snr_std_devs.append(np.std(trial_accuracies))

        perceiver_accuracies.append(perceiver_snr_accuracies)
        perceiver_std_devs.append(perceiver_snr_std_devs)

        # Test my_model (Model_channel)
        config.my_model.name = 'dgm_channel'
        trainer_my_model, my_model, datamodule = setup_training(config)

        my_model_snr_accuracies = []
        my_model_snr_std_devs = []

        for snr in config.exp.test_snr_val:
            trial_accuracies = []

            for _ in range(config.exp.num_trials):
                my_model.snr_db = snr
                test_result = trainer_my_model.validate(my_model, datamodule)
                trial_accuracies.append(test_result[0]['val_acc'])

            my_model_snr_accuracies.append(np.mean(trial_accuracies))
            my_model_snr_std_devs.append(np.std(trial_accuracies))

        my_model_accuracies.append(my_model_snr_accuracies)
        my_model_std_devs.append(my_model_snr_std_devs)

    # Now plot the results for both Perceiver and my_model (Model_channel)
    plot_comparison(perceiver_accuracies, perceiver_std_devs, my_model_accuracies, my_model_std_devs, 
                    config.exp.test_snr_val, config.exp.pooling_ratios, config.pooling.pooling_type, 
                    config.dataset.loader.parameters.data_name, "Simple MLP Bottleneck vs DGM and Graph Pooling")


def plot_comparison(perceiver_accuracies, perceiver_std_devs, my_model_accuracies, my_model_std_devs, 
                    snr_vals, pooling_ratios, pooling_type, data_name, title, save_dir="./plots"):
    
    plt.figure(figsize=(10, 6))
    
    # Define line styles and markers
    perceiver_style = 'o'  
    my_model_style = 'x'   


    # Plot Perceiver and My Model (DGM and Graph Pooling) with the same color for each pooling ratio
    for i, ratio in enumerate(pooling_ratios):
        color = f'C{i}'  # Same color for both Perceiver and My Model for the same ratio

        # Plot Perceiver results
        plt.errorbar(snr_vals, perceiver_accuracies[i], yerr=perceiver_std_devs[i], label=f'MLP Bottleneck - Pooling Ratio: {ratio}',
                     fmt=perceiver_style, linestyle = '-.' , color=color)

        # Plot My Model (DGM and Graph Pooling) results
        plt.errorbar(snr_vals, my_model_accuracies[i], yerr=my_model_std_devs[i], label=f'DGM and Graph Pooling - Pooling Ratio: {ratio}',
                     fmt=my_model_style, linestyle = '-', color=color)

    # Set plot title and labels
    plt.title(f'{title} {pooling_type} on {data_name} Dataset')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)

    # Custom legend with two columns: color for pooling ratio, line style for model type
    # Create legend elements for pooling ratios (colors) and models (line styles)
    legend_elements_color = [Line2D([0], [0], color=f'C{i}', lw=2, label=f'Pooling Ratio: {pooling_ratios[i]}') 
                             for i in range(len(pooling_ratios))]

    legend_elements_style = [Line2D([0], [0], color='black', lw=2, label='MLP Bottleneck', marker='o', ls='-.'),
                             Line2D([0], [0], color='black', lw=2, label='DGM and Graph Pooling', marker='x', ls='-')]

    # Create two-column legend
    first_legend = plt.legend(handles=legend_elements_color, loc='upper left', bbox_to_anchor=(1, 1), title='Pooling Ratios')
    second_legend = plt.legend(handles=legend_elements_style, loc='upper left', bbox_to_anchor=(1, 0.5), title='Model Type')

    # Add both legends to the plot
    plt.gca().add_artist(first_legend)

    # Tight layout to avoid clipping
    plt.tight_layout()

    # Show the plot
    plt.grid(True)
    folder_path = f'mlp_bottleneck_vs_ours/{pooling_type}/{data_name}'
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(f'{folder_path}/topk_dgm.png')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":

    #train_and_plot()
    setup_training()
    #train_and_plot_comparison()

