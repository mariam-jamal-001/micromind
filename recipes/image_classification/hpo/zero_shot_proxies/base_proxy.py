from abc import ABC, abstractmethod

class BaseProxy(ABC):
    @abstractmethod
    def compute(self, model, data_loader):
        """
        Compute the proxy score for a given model and dataset.
        Args:
            model: The neural network model to evaluate.
            data_loader: A DataLoader object for the dataset.
        Returns:
            float: The computed proxy score.
        """
        pass
