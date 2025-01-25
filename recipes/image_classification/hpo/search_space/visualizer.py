import matplotlib.pyplot as plt
import numpy as np
import os

class HyperparameterVisualizer:
    """
    A class to visualize the distributions of sampled hyperparameters.
    """

    def __init__(self, output_dir):
        """
        Initialize with the directory to save plots.

        Args:
            output_dir (str): Directory to save plots and analysis results.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_histogram(self, data, bins=50, title="Distribution", xlabel="Value", ylabel="Frequency", filename="histogram.jpg"):
        """
        Plot a histogram of data.

        Args:
            data (list or numpy.ndarray): Data to plot.
            bins (int): Number of bins.
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            filename (str): Filename to save the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        save_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_scatter(self, x, y, xlabel, ylabel, title, filename="scatter.jpg"):
        """
        Plot a scatter plot of two variables.

        Args:
            x (list or numpy.ndarray): X-axis data.
            y (list or numpy.ndarray): Y-axis data.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            title (str): Plot title.
            filename (str): Filename to save the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.6)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        save_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
