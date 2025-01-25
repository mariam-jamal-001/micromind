import micromind as mm
from micromind.utils import parse_configuration
from recipes.image_classification.train import ImageClassification
from recipes.image_classification.prepare_data import create_loaders
from argparse import Namespace

class PhiNetModelInitializer:
    """
    A class to initialize the PhiNet model based on sampled hyperparameters.
    """

    def __init__(self, config_file_path):
        """
        Initialize with the path to the configuration file.

        Args:
            config_file_path (str): Path to the PhiNet configuration file.
        """
        self.config_file_path = config_file_path

    def initialize_model(self, sampled_hyperparameters):
        """
        Initialize PhiNet model based on sampled hyperparameters.

        Args:
            sampled_hyperparameters (dict): A dictionary containing sampled hyperparameters.

        Returns:
            model (torch.nn.Module): Initialized PhiNet model.
            loaders (tuple): Training and validation data loaders.
        """
        # Update the configuration file with the sampled hyperparameters
        hparams = parse_configuration(self.config_file_path)
        hparams_dict = vars(hparams)

        # Update the configuration dictionary with the sampled hyperparameters
        hparams_dict.update(sampled_hyperparameters)

        # Convert the updated dictionary back to Namespace
        hparams = Namespace(**hparams_dict)

        # Initialize the model using the updated configuration
        mind = ImageClassification(hparams=hparams).modules["classifier"] 
        
        loaders = create_loaders(hparams)

        # Return the model and data loaders
        return mind, loaders

    @staticmethod
    def get_phinet_params(model):
        """
        Get the number of parameters in the PhiNet model.

        Args:
            model (torch.nn.Module): Initialized PhiNet model.

        Returns:
            int: Number of parameters.
        """
        return model.get_params()

    @staticmethod
    def get_phinet_macs(model):
        """
        Get the MAC (Multiply-Accumulate Operations) of the PhiNet model.

        Args:
            model (torch.nn.Module): Initialized PhiNet model.

        Returns:
            int: Number of MACs.
        """
        return model.get_MAC()
