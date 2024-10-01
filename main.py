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
from baseline_models import MLP_KMeans, MLP_PCA, Perceiver_channel, MLP_Bottleneck, Knn_channel
import pickle

@hydra.main(version_base=None, config_path="conf", config_name="config")
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

    if config.my_model.name == 'dgm_channel':
        channel = Model_channel(hparams)
        trainer = pl.Trainer(max_epochs=config.training.max_epochs, accelerator = "cpu", callbacks=[checkpoint_callback, early_stopping_callback])#, logger=wandb_logger, log_every_n_steps=2)
        trainer.fit(channel, datamodule)
        best_model_path = checkpoint_callback.best_model_path
        best_model = Model_channel.load_from_checkpoint(best_model_path, hparams=hparams)

    elif config.my_model.name == 'perceiver': 
        channel = Perceiver_channel(hparams)
        trainer = pl.Trainer(max_epochs=config.training.max_epochs, accelerator = "cpu", callbacks=[checkpoint_callback, early_stopping_callback])#, logger=wandb_logger, log_every_n_steps=2)
        trainer.fit(channel, datamodule)
        best_model_path = checkpoint_callback.best_model_path
        best_model = Perceiver_channel.load_from_checkpoint(best_model_path, hparams=hparams) 

    elif config.my_model.name == 'mlp_bottleneck': 
        channel = MLP_Bottleneck(hparams)
        trainer = pl.Trainer(max_epochs=config.training.max_epochs, accelerator = "cpu", callbacks=[checkpoint_callback, early_stopping_callback],)# logger=wandb_logger, log_every_n_steps=2)
        trainer.fit(channel, datamodule)
        best_model_path = checkpoint_callback.best_model_path
        best_model = MLP_Bottleneck.load_from_checkpoint(best_model_path, hparams=hparams) 

    elif config.my_model.name == 'Knn_channel': 
        channel = Knn_channel(hparams)
        trainer = pl.Trainer(max_epochs=config.training.max_epochs, accelerator = "cpu", callbacks=[checkpoint_callback, early_stopping_callback],)# logger=wandb_logger, log_every_n_steps=2)
        trainer.fit(channel, datamodule)
        best_model_path = checkpoint_callback.best_model_path
        best_model = Knn_channel.load_from_checkpoint(best_model_path, hparams=hparams) 


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

        # Test baseline model
        config.my_model.name = 'perceiver'
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
                    config.dataset.loader.parameters.data_name, "Perceiver vs DGM and Graph Pooling")


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
    legend_elements_color = [Line2D([0], [0], color=f'C{i}', lw=2, label=f'{pooling_ratios[i]}') 
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
    folder_path = f'perceiver_vs_ours/{pooling_type}/{data_name}'
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(f'{folder_path}/topk_dgm.png')
    plt.tight_layout()
    plt.show()







# for fixed SNR in validation, compare the different pooling methods

def compare_poolings(config: DictConfig, snr):

    config.my_model.channel.snr_db = snr
    results = {}

    for pool_method in config.exp.pool_methods: 

        config.pooling.pooling_type = pool_method
        method_accuracies = []
        method_std = []

        if config.pooling.pooling_type == 'perceiver': 
            config.my_model.name = 'perceiver'
        elif config.pooling.pooling_type == 'mlp_bottleneck':
            config.my_model.name = 'perceiver'
        elif config.pooling.pooling_type in ['asa', 'sag', 'topk']:
            config.my_model.name = 'dgm_channel'


        for pool_ratio in config.exp.pooling_ratios:

            config.pooling.pooling_ratio = pool_ratio
            trainer, model, datamodule = setup_training(config)
            trial_accuracies = []

            for _ in range(config.exp.num_trials):

                test_result = trainer.validate(model, datamodule)
                trial_accuracies.append(test_result[0]['val_acc'])

            method_accuracies.append(np.mean(trial_accuracies))
            method_std.append(np.std(trial_accuracies))

        results[pool_method] = {
            "accuracies": method_accuracies,
            "std": method_std
        }

    plot_results_pool(results, config.exp.pooling_ratios, config)


def plot_results_pool(results, pooling_ratios, config):
    plt.figure(figsize=(10, 6))

    for pool_method, results in results.items():

        accuracies = results["accuracies"]
        std_dev = results["std"]
        plt.errorbar(pooling_ratios, accuracies, yerr=std_dev, label=f"{pool_method.upper()}", capsize=5, marker='o')

    plt.title(f"Accuracy vs Pooling Ratio for Different Pooling Methods. Fixed SNR to {config.my_model.channel.snr_db}")
    plt.xlabel("Pooling Ratio")
    plt.ylabel("Accuracy")
    plt.legend(title="Pooling Methods")
    plt.grid(True)
    plt.tight_layout()


# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def compare_poolings_fixed_snr(config: DictConfig):
#     save_dir = "compare_poolings_fixedsnr" 
#     os.makedirs(save_dir, exist_ok=True)  

#     for snr_value in config.exp.snr_values:
        
#         print(f"Running experiment with SNR: {snr_value} dB")

#         compare_poolings(config, snr_value)  
#         filename = os.path.join(save_dir, f"accuracy_vs_pooling_snr_{snr_value}.png")
#         plt.savefig(filename)
#         plt.close()  

#         print(f"Plot saved to {filename}")





def save_results(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")

def load_results(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        print(f"Results loaded from {filename}")
        return results
    return {}


def compare_poolings(config: DictConfig, trainer, snr, model, datamodule, pool_method, pool_ratio, results):
    """
    Compare model validation performance for a fixed SNR value and different pooling methods and ratios.
    """
    model.snr_db = snr

    trial_accuracies = []


    # Validate the trained model multiple times for stability
    for _ in range(config.exp.num_trials):
        test_result = trainer.validate(model, datamodule)
        trial_accuracies.append(test_result[0]['val_acc'])

    if snr not in results:
        results[snr] = {}

    if pool_method not in results[snr]:
        results[snr][pool_method] = {}

    results[snr][pool_method][pool_ratio] = {
        "accuracies": np.mean(trial_accuracies),
        "std": np.std(trial_accuracies)
    }

    return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def compare_poolings_fixed_snr(config: DictConfig):
    save_dir = "new_dgm/results_poolings_snrs_without_noise/imdb"
    os.makedirs(save_dir, exist_ok=True)
    results_file = os.path.join(save_dir, "results.pkl")

    results = load_results(results_file)

    # Loop over each pooling method and pooling ratio (these require retraining)
    for pool_method in config.exp.pool_methods:

        config.pooling.pooling_type = pool_method

        if config.pooling.pooling_type == 'perceiver': 
            config.my_model.name = 'perceiver'
        elif config.pooling.pooling_type == 'mlp_bottleneck':
            config.my_model.name = 'mlp_bottleneck'
        elif config.pooling.pooling_type in ['asa', 'sag', 'topk']:
            config.my_model.name = 'dgm_channel'

        for pool_ratio in config.exp.pooling_ratios:

            config.pooling.pooling_ratio = pool_ratio

            # Train the model from scratch for the current pooling type and ratio
            print(f"Training model with Pooling Method: {pool_method}, Pooling Ratio: {pool_ratio}")
            trainer, best_model, datamodule = setup_training(config)

            # Validate the trained model for different SNR values
            for snr_value in config.exp.snr_values:
                print(f"Validating with SNR: {snr_value} dB")

                # Validate the model with the current SNR value
                results = compare_poolings(config, trainer, snr_value, best_model, datamodule, pool_method, pool_ratio, results)

    # Generate a plot for this SNR value
    save_results(results, results_file)
    plot_results_pool_per_snr(results, config.exp.pooling_ratios, config)
    plot_results_pool_per_ratio(results, config.exp.snr_values, config)

    print(f"Results saved to {save_dir}")




def plot_results_pool_per_snr(results, pooling_ratios, config):
    # Iterate over each SNR value in the results dictionary

    ylims = [(0.45, 0.76), (0.54, 0.755), (0.67, 0.755), (0.69, 0.755), (0.685, 0.755)]
    for s, snr_value in enumerate(results):
        plt.figure(figsize=(10, 6))
        
        # Get the list of pooling methods for easy access
        pool_methods = list(results[snr_value].keys())
        
        # Store line styles and labels for the legend
        legend_lines = []
        legend_labels = []

        # Iterate over each pooling method for the given SNR value
        for idx, (pool_method, pool_data) in enumerate(results[snr_value].items()):
            accuracies = []
            std_devs = []

            # Extract accuracies and std deviations for each pooling ratio in the current method
            for pool_ratio in pooling_ratios:
                if pool_ratio in pool_data:
                    accuracies.append(pool_data[pool_ratio]["accuracies"])
                    std_devs.append(pool_data[pool_ratio]["std"])
                else:
                    print(f"No data found for pooling ratio: {pool_ratio} under method: {pool_method}")
                    accuracies.append(None)
                    std_devs.append(0)

            # Determine line style and properties
            line_style = '--' if idx >= len(pool_methods) - 2 else '-'  # Dashed line for the last two methods
            line, _, _ = plt.errorbar(
                pooling_ratios, accuracies, yerr=std_devs, 
                label=f"{pool_method.upper()}", capsize=5, marker='o', 
                linestyle=line_style, linewidth=2.5, alpha=0.7  # Thicker lines and reduced opacity
            )

            # Append Line2D object and label for legend
            legend_lines.append(Line2D([0], [0], color=line.get_color(), linestyle=line_style, linewidth=2.5))
            legend_labels.append(pool_method.upper())

        # Customize each plot for the current SNR value
        # plt.ylim(ylims[s])
        plt.title(f"Accuracy vs Pooling Ratio for SNR: {snr_value} dB", fontsize=16)  # Increased font size
        plt.xlabel("Pooling Ratio", fontsize=16)  # Increased font size
        plt.ylabel("Accuracy", fontsize=16)  # Increased font size
        plt.xticks(fontsize=14)  # Increased tick size
        plt.yticks(fontsize=14)  # Increased tick size

        # Use the legend_lines and legend_labels to create the legend
        plt.legend(legend_lines, legend_labels, fontsize=12)
        
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as a PNG file
        save_dir = "new_dgm/comparison_plots/imdb/with_without_dgm/compare_poolings_snr_plots"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"accuracy_vs_pooling_ratio_snr_{snr_value}.png")
        plt.savefig(filename)
        plt.close()  # Close the plot to avoid overlap

        print(f"Plot saved to {filename}")

def plot_results_pool_per_ratio(results, snr_values, config):
    ylims = [(0.454, 0.76), (0.62, 0.76), (0.62, 0.76), (0.574, 0.75), (0.61, 0.75)]
    # Iterate over each pooling ratio
    for p, pool_ratio in enumerate(config.exp.pooling_ratios):
        plt.figure(figsize=(10, 6))
        
        # Get the list of pooling methods for easy access
        pool_methods = config.exp.pool_methods

        # Store line styles and labels for the legend
        legend_lines = []
        legend_labels = []
        
        # Iterate over each pooling method
        for idx, pool_method in enumerate(pool_methods):
            accuracies = []
            std_devs = []

            # Collect accuracy and std dev for each SNR value for this pooling ratio
            for snr_value in snr_values:
                if pool_method in results[snr_value] and pool_ratio in results[snr_value][pool_method]:
                    accuracies.append(results[snr_value][pool_method][pool_ratio]["accuracies"])
                    std_devs.append(results[snr_value][pool_method][pool_ratio]["std"])
                else:
                    print(f"No data found for SNR: {snr_value}, Method: {pool_method}, Ratio: {pool_ratio}")
                    accuracies.append(None)
                    std_devs.append(0)

            # Determine line style and properties
            line_style = '--' if idx >= len(pool_methods) - 2 else '-'  # Dashed line for the last two methods
            line, _, _ = plt.errorbar(
                snr_values, accuracies, yerr=std_devs, label=f"{pool_method.upper()}", 
                capsize=5, marker='o', linestyle=line_style, 
                linewidth=2.5, alpha=0.7  # Thicker lines and reduced opacity
            )

            # Append Line2D object and label for legend
            legend_lines.append(Line2D([0], [0], color=line.get_color(), linestyle=line_style, linewidth=2.5))
            legend_labels.append(pool_method.upper())

        # Customize each plot for the current pooling ratio
        # plt.ylim(ylims[p])
        plt.title(f"Accuracy vs SNR for Pooling Ratio: {pool_ratio}", fontsize=16)  # Increased font size
        plt.xlabel("SNR (dB)", fontsize=14)  # Increased font size
        plt.ylabel("Accuracy", fontsize=14)  # Increased font size
        plt.xticks(fontsize=12)  # Increased tick size
        plt.yticks(fontsize=12)  # Increased tick size
        
        # Use the legend_lines and legend_labels to create the legend
        plt.legend(legend_lines, legend_labels, fontsize=12)
        
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as a PNG file
        save_dir = "new_dgm/comparison_plots/imdb/with_without_dgm/compare_poolings_ratio_plots"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"accuracy_vs_snr_ratio_{pool_ratio}.png")
        plt.savefig(filename)
        plt.close()  # Close the plot to avoid overlap

        print(f"Plot saved to {filename}")




@hydra.main(version_base=None, config_path="conf", config_name="config")
def plot_existing_res(config: DictConfig):
    filename = 'results_with_without_dgm/imdb/results.pkl'
    results = load_results(filename)
    results['dgm'] = load_results('results.pkl')
    plot_results_pool_per_snr(results, config.exp.pooling_ratios, config)
    plot_results_pool_per_ratio(results, config.exp.snr_values, config)



@hydra.main(version_base=None, config_path="conf", config_name="config")
def compare_with_without_dgm(config: DictConfig):

    results = { 'dgm': {}, 'no_dgm': {}}

    results['dgm'] = load_results('new_dgm\results_poolings_snrs_with_noise\imdb\results.pkl')

    config.training.noisy = True
    # ensure training with noise and with dgm
    config.dgm.name = 'no_dgm'

    # loop over different pooling types (ASA, SAG, TopK), train and evaluate without dgm
    for pooling_type in config.exp.pool_methods:
        config.pooling.pooling_type = pooling_type

        for pooling_ratio in config.exp.pooling_ratios: 
            config.pooling.pooling_ratio = pooling_ratio 

            # Train the model from scratch for the current pooling type and ratio
            print(f"Training model with Pooling Method: {pooling_type}, Pooling Ratio: {pooling_ratio}")
            trainer, best_model, datamodule = setup_training(config)

            # Validate the trained model for different SNR values
            for snr_value in config.exp.snr_values:
                print(f"Validating with SNR: {snr_value} dB")

                # Validate the model with the current SNR value
                results['no_dgm'] = compare_poolings(config, trainer, snr_value, best_model, datamodule, pooling_type, pooling_ratio, results['no_dgm'])


    save_dir = "results_with_without_dgm/imdb"
    os.makedirs(save_dir, exist_ok=True)
    results_file = os.path.join(save_dir, "results.pkl")
    save_results(results, results_file)

    return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def plot_results_per_pooling_method(config):

    filename = 'results_with_without_dgm/imdb/results.pkl'  
    results = load_results(filename)
    results['dgm'] = load_results('results.pkl')
    # Iterate over each pooling method (ASA, SAG, TopK, etc.)
    for pool_method in config.exp.pool_methods:
        # Iterate over each pooling ratio
        for pool_ratio in config.exp.pooling_ratios:
            plt.figure(figsize=(10, 6))

            # Store line styles and labels for the legend
            legend_lines = []
            legend_labels = []

            accuracies_with_dgm = []
            std_devs_with_dgm = []
            accuracies_without_dgm = []
            std_devs_without_dgm = []

            # Collect accuracy and std dev for each SNR value for this pooling ratio
            for snr_value in config.exp.snr_values:
                # Collect data for DGM
                if pool_method in results['dgm'][snr_value] and pool_ratio in results['dgm'][snr_value][pool_method]:
                    accuracies_with_dgm.append(results['dgm'][snr_value][pool_method][pool_ratio]["accuracies"])
                    std_devs_with_dgm.append(results['dgm'][snr_value][pool_method][pool_ratio]["std"])
                else:
                    accuracies_with_dgm.append(None)
                    std_devs_with_dgm.append(0)

                # Collect data for no DGM
                if pool_method in results['no_dgm'][snr_value] and pool_ratio in results['no_dgm'][snr_value][pool_method]:
                    accuracies_without_dgm.append(results['no_dgm'][snr_value][pool_method][pool_ratio]["accuracies"])
                    std_devs_without_dgm.append(results['no_dgm'][snr_value][pool_method][pool_ratio]["std"])
                else:
                    accuracies_without_dgm.append(None)
                    std_devs_without_dgm.append(0)

            # Plotting DGM results
            line_with_dgm, _, _ = plt.errorbar(
                config.exp.snr_values, accuracies_with_dgm, yerr=std_devs_with_dgm, label=f"DGM", 
                capsize=5, marker='o', linestyle='-', linewidth=2.5, alpha=0.7
            )
            # Plotting no DGM results
            line_without_dgm, _, _ = plt.errorbar(
                config.exp.snr_values, accuracies_without_dgm, yerr=std_devs_without_dgm, label=f"No DGM", 
                capsize=5, marker='s', linestyle='--', linewidth=2.5, alpha=0.7
            )
            plt.ylim(0.60, 0.76)
            # Append Line2D object and label for legend
            legend_lines.append(Line2D([0], [0], color=line_with_dgm.get_color(), linestyle='-', linewidth=2.5))
            legend_labels.append(f"DGM")
            legend_lines.append(Line2D([0], [0], color=line_without_dgm.get_color(), linestyle='--', linewidth=2.5))
            legend_labels.append(f"No DGM")

            # Customize each plot for the current pooling method and ratio
            plt.title(f"Accuracy vs SNR for {pool_method.upper()} Pooling Ratio: {pool_ratio}", fontsize=16)
            plt.xlabel("SNR (dB)", fontsize=14)
            plt.ylabel("Accuracy", fontsize=14)
            plt.xticks(config.exp.snr_values, fontsize=12)
            plt.yticks(fontsize=12)

            # Use the legend_lines and legend_labels to create the legend
            plt.legend(legend_lines, legend_labels, title="Model", fontsize=12)

            # Add grid and layout
            plt.grid(True)
            plt.tight_layout()

            # Save the plot as a PNG file
            save_dir = f"comparison_plots/imdb/with_without_dgm/{pool_method}"
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f"accuracy_vs_snr_{pool_method}_ratio_{pool_ratio}.png")
            plt.savefig(filename)
            plt.close()  # Close the plot to avoid overlap

            print(f"Plot saved to {filename}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def plot_results_comparison(config):
    """
    Generates and saves plots for different pooling methods and pooling ratios, comparing with and without noise data.
    
    Args:
        with_noise_results: Results with noise.
        without_noise_results: Results without noise.
        snr_values: A list of SNR values.
        config: The configuration object that contains pool methods and ratios.
    """
    with_noise_results = load_results('with_noise_results.pkl')
    without_noise_results = load_results('without_noise_results.pkl')
    snr_values = config.exp.snr_values 
    # Define styles and figure settings
    line_styles = ['-', '--']  # Solid for without noise, dashed for with noise
    ylims = [(0.454, 0.76), (0.62, 0.76), (0.62, 0.76), (0.574, 0.75), (0.61, 0.75)]
    
    # Get the list of pooling methods and ratios
    pool_methods = config.exp.pool_methods
    pooling_ratios = config.exp.pooling_ratios
    
    # Loop over each pooling ratio and method
    for p, pool_ratio in enumerate(pooling_ratios):
        for idx, pool_method in enumerate(pool_methods):
            plt.figure(figsize=(10, 6))  # Create a new figure for each combination

            # Initialize lists for accuracies and std deviations for both conditions
            accuracies_with_noise, std_devs_with_noise = [], []
            accuracies_without_noise, std_devs_without_noise = [], []

            # Collect accuracy and std dev for each SNR value for both conditions
            for snr_value in snr_values:
                # With noise
                if pool_method in with_noise_results[snr_value] and pool_ratio in with_noise_results[snr_value][pool_method]:
                    accuracies_with_noise.append(with_noise_results[snr_value][pool_method][pool_ratio]["accuracies"])
                    std_devs_with_noise.append(with_noise_results[snr_value][pool_method][pool_ratio]["std"])
                else:
                    print(f"No data found (with noise) for SNR: {snr_value}, Method: {pool_method}, Ratio: {pool_ratio}")
                    accuracies_with_noise.append(None)
                    std_devs_with_noise.append(0)

                # Without noise
                if pool_method in without_noise_results[snr_value] and pool_ratio in without_noise_results[snr_value][pool_method]:
                    accuracies_without_noise.append(without_noise_results[snr_value][pool_method][pool_ratio]["accuracies"])
                    std_devs_without_noise.append(without_noise_results[snr_value][pool_method][pool_ratio]["std"])
                else:
                    print(f"No data found (without noise) for SNR: {snr_value}, Method: {pool_method}, Ratio: {pool_ratio}")
                    accuracies_without_noise.append(None)
                    std_devs_without_noise.append(0)

            # Plot with noise
            line_with_noise, _, _ = plt.errorbar(
                snr_values, accuracies_with_noise, yerr=std_devs_with_noise, label=f"{pool_method.upper()} (With Noise)", 
                capsize=5, marker='o', linestyle=line_styles[1],  # Dashed line for with noise
                linewidth=2.5, alpha=0.7  # Match style with opacity and line thickness
            )
            
            # Plot without noise
            line_without_noise, _, _ = plt.errorbar(
                snr_values, accuracies_without_noise, yerr=std_devs_without_noise, label=f"{pool_method.upper()} (Without Noise)", 
                capsize=5, marker='o', linestyle=line_styles[0],  # Solid line for without noise
                linewidth=2.5, alpha=0.7
            )
            
            # Set the limits for the y-axis dynamically if needed
            # plt.ylim(ylims[p])

            # Title and labels with increased font size
            plt.title(f"Accuracy vs SNR for Pooling Ratio: {pool_ratio}, Method: {pool_method.upper()}", fontsize=16)
            plt.xlabel("SNR (dB)", fontsize=14)
            plt.ylabel("Accuracy", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            
            # Add legend
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()

            # Save the plot for this specific combination of ratio and method
            save_dir = f"new_dgm/comparison_plots/imdb/with_without_dgm/compare_poolings_ratio_plots"
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f"accuracy_vs_snr_ratio_{pool_ratio}_method_{pool_method}.png")
            plt.savefig(filename)
            plt.close()  # Close the plot to avoid memory issues

            print(f"Plot saved to {filename}")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def plot_gap_comparison_for_methods(config):
    """
    Generates bar charts to compare the accuracy gap across different methods for each pooling ratio.
    Each bar represents a method, and the height corresponds to the accuracy gap for a given SNR.
    
    Args:
        with_noise_results: Results with noise.
        without_noise_results: Results without noise.
        snr_values: A list of SNR values.
        config: The configuration object containing pool methods and ratios.
    """
    with_noise_results = load_results('with_noise_results.pkl')
    without_noise_results = load_results('without_noise_results.pkl')
    snr_values = config.exp.snr_values 
    # Get pooling methods and ratios from the config
    pool_methods = config.exp.pool_methods
    pooling_ratios = config.exp.pooling_ratios
    
    # Define bar width and positions
    bar_width = 0.15  # Width of each bar
    n_methods = len(pool_methods)
    
    # Loop over each pooling ratio
    for pool_ratio in pooling_ratios:
        plt.figure(figsize=(12, 6))  # Create a new figure for each pooling ratio

        # Store the bar positions for each method for a given SNR value
        indices = np.arange(len(snr_values))  # Number of bars (one per SNR)

        # For each pooling method, calculate the gap and plot the bars
        for idx, pool_method in enumerate(pool_methods):
            accuracy_gaps = []

            # Collect the accuracy gap for each SNR
            for snr_value in snr_values:
                if (pool_method in with_noise_results[snr_value] and pool_ratio in with_noise_results[snr_value][pool_method]
                    and pool_method in without_noise_results[snr_value] and pool_ratio in without_noise_results[snr_value][pool_method]):
                    
                    accuracy_with_noise = with_noise_results[snr_value][pool_method][pool_ratio]["accuracies"]
                    accuracy_without_noise = without_noise_results[snr_value][pool_method][pool_ratio]["accuracies"]

                    # Calculate the accuracy gap
                    accuracy_gap = accuracy_with_noise - accuracy_without_noise
                    accuracy_gaps.append(accuracy_gap)
                else:
                    print(f"No data found for SNR: {snr_value}, Method: {pool_method}, Ratio: {pool_ratio}")
                    accuracy_gaps.append(0)  # Assign zero if data not available

            # Plot the bars for the current pooling method
            plt.bar(indices + idx * bar_width, accuracy_gaps, bar_width, label=pool_method.upper())

        # Customize the plot
        plt.title(f"Accuracy Gap Comparison for Pooling Ratio: {pool_ratio}", fontsize=16)
        plt.xlabel("SNR (dB)", fontsize=14)
        plt.ylabel("Accuracy Gap (With Noise - Without Noise)", fontsize=14)
        plt.xticks(indices + bar_width * (n_methods - 1) / 2, snr_values, fontsize=12)  # Center ticks between bars
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, axis='y')

        # Save the plot
        save_dir = f"new_dgm/comparison_plots/imdb/gap_comparison_plots"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"gap_comparison_ratio_{pool_ratio}.png")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        print(f"Gap comparison plot saved to {filename}")



if __name__ == "__main__":

    
    # setup_training()
    
    # compare_poolings_fixed_snr()
    # plot_existing_res()
    # compare_with_without_dgm()
    # compare_with_without_dgm()
    # plot_results_per_pooling_method()
    # plot_results_comparison()
    plot_gap_comparison_for_methods()
    