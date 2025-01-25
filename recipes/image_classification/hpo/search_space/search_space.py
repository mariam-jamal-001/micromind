import logging
import os
from recipes.image_classification.hpo.search_space.sampler import HyperparameterSampler
from recipes.image_classification.hpo.phinet_model import PhiNetModelInitializer
from recipes.image_classification.hpo.search_space.visualizer import HyperparameterVisualizer

class SearchSpace:
    """
    A class to define, sample, and analyze a search space for neural architecture search, including PhiNet models.
    """

    def __init__(self, output_dir="output", sampler_type="random", config_file_path="../../cfg/phinet.py"):
        """
        Initialize the search space.

        Args:
            output_dir (str): Directory to save plots and analysis results.
            sampler_type (str): Sampling algorithm to use ('random', 'tpe', etc.).
            config_file_path (str): Path to the PhiNet configuration file.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize components
        self.sampler = HyperparameterSampler(sampler_type)
        self.phinet_initializer = PhiNetModelInitializer(config_file_path)
        self.visualizer = HyperparameterVisualizer(self.output_dir)
        logging.info(f"SearchSpace initialized with {sampler_type} sampler.")

    def sample_architectures(self, num_samples=100):
        """
        Sample architectures from the search space.

        Args:
            num_samples (int): Number of architectures to sample.

        Returns:
            list: A list of sampled hyperparameters.
        """
        samples = []
        logging.info(f"Sampling {num_samples} architectures...")
        for _ in range(num_samples):
            trial = self.sampler.study.ask()
            sample = self.sampler.sample(trial)

            # Initialize the model and get number of parameters
            model, _ = self.phinet_initializer.initialize_model(sample)
            sample["num_params"] = self.phinet_initializer.get_phinet_params(model)
            sample["num_macs"] = self.phinet_initializer.get_phinet_macs(model)

            samples.append(sample)
        logging.info("Sampling completed.")

        return samples

    def initialize_phinet_model(self, sampled_hyperparameters):
        """
        Initialize PhiNet model based on sampled hyperparameters.

        Args:
            sampled_hyperparameters (dict): A dictionary containing sampled hyperparameters.

        Returns:
            model (torch.nn.Module): Initialized PhiNet model.
            loaders (tuple): Training and validation data loaders.
        """
        return self.phinet_initializer.initialize_model(sampled_hyperparameters)

    def analyze_search_space(self, samples):
        """
        Analyze the search space by visualizing the sampled data.

        Args:
            samples (list): List of sampled hyperparameter dictionaries.
        """
        alpha_vals = [s["alpha"] for s in samples]
        beta_vals = [s["beta"] for s in samples]
        t_zero_vals = [s["t_zero"] for s in samples]
        num_layers_vals = [s["num_layers"] for s in samples]
        num_params_vals = [s["num_params"] for s in samples]
        num_mac_vals = [s["num_macs"] for s in samples]

        

        self.visualizer.plot_histogram(alpha_vals, title="Alpha Distribution", xlabel="Alpha", filename="alpha_histogram.jpg")
        self.visualizer.plot_histogram(beta_vals, title="Beta Distribution", xlabel="Beta", filename="beta_histogram.jpg")
        self.visualizer.plot_histogram(t_zero_vals, title="T_zero Distribution", xlabel="T_zero", filename="t_zero_histogram.jpg")
        self.visualizer.plot_histogram(num_layers_vals, title="Num Layers Distribution", xlabel="Num Layers", filename="num_layers_histogram.jpg")
        self.visualizer.plot_histogram(num_params_vals, title="Num Params Distribution", xlabel="#Params", filename="num_params_histogram.jpg")
        self.visualizer.plot_histogram(num_mac_vals, title="Num MACs Distribution", xlabel="#MACs", filename="num_macs_histogram.jpg")

        self.visualizer.plot_scatter(alpha_vals, num_params_vals, "Alpha", "Num Params", "Alpha vs Num Params", filename="alpha_vs_num_params.jpg")
        self.visualizer.plot_scatter(beta_vals, num_params_vals, "Beta", "Num Params", "Beta vs Num Params", filename="beta_vs_num_params.jpg")
        self.visualizer.plot_scatter(t_zero_vals, num_params_vals, "T_zero", "Num Params", "T_zero vs Num Params", filename="t_zero_vs_num_params.jpg")
        self.visualizer.plot_scatter(num_layers_vals, num_params_vals, "Num Layers", "Num Params", "Num Layers vs Num Params", filename="num_layers_vs_num_params.jpg")

        self.visualizer.plot_scatter(alpha_vals, num_mac_vals, "Alpha", "Num Macs", "Alpha vs Num Macs", filename="alpha_vs_num_macs.jpg")
        self.visualizer.plot_scatter(beta_vals, num_mac_vals, "Beta", "Num Macs", "Beta vs Num Macs", filename="beta_vs_num_macs.jpg")
        self.visualizer.plot_scatter(t_zero_vals, num_mac_vals, "T_zero", "Num Macs", "T_zero vs Num Macs", filename="t_zero_vs_num_macs.jpg")
        self.visualizer.plot_scatter(num_layers_vals, num_mac_vals, "Num Layers", "Num Macs", "Num Layers vs Num Macs", filename="num_layers_vs_num_macs.jpg")

if __name__ == "__main__":
    search_space = SearchSpace(output_dir="output/search_space")
    samples = search_space.sample_architectures(num_samples=100)
    search_space.analyze_search_space(samples)
    