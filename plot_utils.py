import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_gcn_heatmap(results):
    # Prepare the data for the heatmap
    pool_methods = ['asa', 'sag', 'topk']
    pooling_ratios = [0.1, 0.25, 0.5, 0.75, 0.99]
    snr_values = [-10, -5, 0, 5, 10]

    # Iterate over each pooling ratio
    for p, pool_ratio in enumerate(pooling_ratios):
        # Create a matrix to store the accuracy differences for each method and SNR
        diff_matrix = np.zeros((len(pool_methods), len(snr_values)))
        
        # Fill the matrix with the differences (GCN - No GCN)
        for i, pool_method in enumerate(pool_methods):
            for j, snr in enumerate(snr_values):
                if (pool_method in results['with_gcn'][snr] and pool_ratio in results['with_gcn'][snr][pool_method] and 
                    pool_method in results['without_gcn'][snr] and pool_ratio in results['without_gcn'][snr][pool_method]):
                    acc_with_gcn = results['with_gcn'][snr][pool_method][pool_ratio]['accuracies']
                    acc_without_gcn = results['without_gcn'][snr][pool_method][pool_ratio]['accuracies']
                    diff_matrix[i, j] = acc_with_gcn - acc_without_gcn
                else:
                    diff_matrix[i, j] = np.nan  # Handle missing data if any

        # Plot the heatmap using seaborn
        plt.figure(figsize=(10, 6))
        sns.heatmap(diff_matrix, annot=True, cmap="coolwarm", xticklabels=snr_values, yticklabels=pool_methods, cbar_kws={'label': 'Δ Accuracy (GCN - No GCN)'})
        
        plt.title(f"GCN vs No GCN - Accuracy Difference Heatmap (Pooling Ratio: {pool_ratio})", fontsize=16)
        plt.xlabel("SNR (dB)", fontsize=14)
        plt.ylabel("Pooling Methods", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        
        # Show or save the plot
        plt.show()


def plot_gcn_difference(results):
    pool_methods = ['asa', 'sag', 'topk']
    pooling_ratios = [0.1, 0.25, 0.5, 0.75, 0.99]
    snr_values = [-10, -5, 0, 5, 10]

    for p, pool_ratio in enumerate(pooling_ratios):
        plt.figure(figsize=(10, 6))
        pool_methods = pool_methods
        for pool_method in pool_methods:
            diff_accuracies = []
            for snr in snr_values:
                if (pool_method in results['with_gcn'][snr] and pool_ratio in results['with_gcn'][snr][pool_method] and 
                    pool_method in results['without_gcn'][snr] and pool_ratio in results['without_gcn'][snr][pool_method]):
                    acc_with_gcn = results['with_gcn'][snr][pool_method][pool_ratio]['accuracies']
                    acc_without_gcn = results['without_gcn'][snr][pool_method][pool_ratio]['accuracies']
                    diff_accuracies.append(acc_with_gcn - acc_without_gcn)
                else:
                    diff_accuracies.append(0)
            
            # Plot difference in accuracies
            plt.plot(snr_values, diff_accuracies, label=f"{pool_method.upper()} (Δ Accuracy)", marker='o')
        
        plt.title(f"Difference in Accuracy (with GCN - without GCN) for Pooling Ratio: {pool_ratio}")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Δ Accuracy")
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example function to plot heatmaps for ablation study
def plot_heatmap_for_methods(results,metric='accuracy'):
    """
    Plots heatmaps comparing the performance (e.g., accuracy) across SNR values and pooling ratios for each pooling method.
    
    Args:
        results: Dictionary with performance results for each pooling method, ratio, and SNR value.
        snr_values: List of SNR values.
        pool_ratios: List of pooling ratios.
        pool_methods: List of pooling methods.
        metric: The performance metric to visualize (default is 'accuracy').
    """

    snr_values, pool_ratios, pool_methods = [-10, -5, 0, 5, 10], [0.1, 0.25, 0.5, 0.75, 1.0], ['asa', 'sag', 'topk'] 
    # Set up the figure for 3 heatmaps (one for each pooling method)
    fig, axes = plt.subplots(1, len(pool_methods), figsize=(20, 6))

    for i, pool_method in enumerate(pool_methods):
        # Prepare data for the heatmap
        heatmap_data = np.zeros((len(snr_values), len(pool_ratios)))
        
        for r, pool_ratio in enumerate(pool_ratios):
            for s, snr_value in enumerate(snr_values):
                if pool_method in results[snr_value] and pool_ratio in results[snr_value][pool_method]:
                    heatmap_data[s, r] = results[snr_value][pool_method][pool_ratio][metric]
                else:
                    heatmap_data[s, r] = np.nan  # Set NaN if data is missing

        # Plot heatmap
        sns.heatmap(
            heatmap_data, 
            annot=True, fmt=".2f", cmap="coolwarm", 
            xticklabels=pool_ratios, yticklabels=snr_values, 
            ax=axes[i], cbar_kws={'label': metric.capitalize()}
        )
        
        axes[i].set_title(f"Pooling Method: {pool_method.upper()}", fontsize=14)
        axes[i].set_xlabel("Pooling Ratios", fontsize=12)
        axes[i].set_ylabel("SNR Values (dB)", fontsize=12)

    # Adjust layout
    plt.tight_layout()
    plt.show()