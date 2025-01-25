import matplotlib as plt
import logging

class SearchSpaceVisualizer:
    """
    A class to visualize the search space
    """

    def __init__(self):
        """
        Initialize the visualizer
        
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def plot_params_histogram(self, params_vals, save_path, bins=50):
        """
        Plot a histogram of the distribution of the number of parameters.

        Args:
            params_vals (numpy.ndarray): Array of number of parameters from the search space.
            bins (int): Number of bins to use in the histogram.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(params_vals, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Parameters')
        plt.ylabel('Frequency')
        plt.title('Distribution of Number of Parameters in the Search Space')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save and show the histogram
        plt.savefig(save_path)
        self.logger.info("Histogram of number of parameters saved.")
    
    def add_subplot(self, row, col, idx, x, y, x_label, y_label, title):
        plt.subplot(row, col, idx)
        plt.scatter(x, y, alpha=0.6)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid()

    # Visualize the results
    def plot_hyperparameters_correlation(self, alpha_vals, beta_vals, t_zero_vals, num_layers_vals, params_vals, mac_vals, save_path):
        plt.figure(figsize=(20, 6))
        
        # alpha vs num_params
        self.add_subplot(2, 4, 1, alpha_vals, params_vals, 'Alpha', 'Number of Parameters (M)', 'Alpha vs Number of Parameters (M)')
        
        # beta vs num_params
        self.add_subplot(2, 4, 2, beta_vals, params_vals, 'Beta', 'Number of Parameters (M)', 'Beta vs Number of Parameters (M)')

        # t_zero vs num_params
        self.add_subplot(2, 4, 3, t_zero_vals, params_vals, 'T_zero', 'Number of Parameters (M)', 'T_zero vs Number of Parameters (M)')

        # num_layers vs num_params
        self.add_subplot(2, 4, 4, num_layers_vals, params_vals, 'Num Layers', 'Number of Parameters (M)', 'Num Layers vs Number of Parameters (M)')

        # alpha vs mac
        self.add_subplot(2, 4, 5, alpha_vals, mac_vals, 'Alpha', 'MACs (M))', 'Alpha vs MACs (M)')

        # beta vs mac
        self.add_subplot(2, 4, 6, beta_vals, mac_vals, 'Beta', 'MACs (M))', 'Beta vs MACs (M)')

        # t_zero vs mac
        self.add_subplot(2, 4, 7, t_zero_vals, mac_vals, 'T_zero', 'MACs (M))', 'T_zero vs MACs (M)')

        # num_layers vs mac
        self.add_subplot(2, 4, 7, num_layers_vals, mac_vals, 'Num Layers', 'MACs (M))', 'Num Layers vs MACs (M)')
       
        plt.tight_layout()
        plt.savefig(save_path)
        self.logger.info("Results plotted and saved.")
        

        

    