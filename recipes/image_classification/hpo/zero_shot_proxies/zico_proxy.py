import os
import sys
import torch
import numpy as np
from torch import nn
from .base_proxy import BaseProxy  # Import the abstract base class
from micromind.utils import parse_configuration
from recipes.image_classification.train import ImageClassification
import time

class ZicoProxy(BaseProxy):
    def __init__(self):
        super(ZicoProxy, self).__init__()

    def getgrad(self, model: torch.nn.Module, grad_dict: dict, step_iter=0):
        """
        Collect gradients from the model and store them in the grad_dict.
        """
        if step_iter == 0:
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                    grad_dict[name] = [mod.weight.grad.data.cpu().reshape(-1).numpy()]
        else:
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                    grad_dict[name].append(mod.weight.grad.data.cpu().reshape(-1).numpy())
        return grad_dict

    def calculate_zico(self, grad_dict):
        """
        Calculate the ZICO score based on the gradient information.
        """
        for modname in grad_dict.keys():
            grad_dict[modname] = np.array(grad_dict[modname])
        
        nsr_mean_sum_abs = 0
        for modname in grad_dict.keys():
            nsr_std = np.std(grad_dict[modname], axis=0)
            nonzero_idx = np.nonzero(nsr_std)[0]
            nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
            tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
            if tmpsum != 0:
                nsr_mean_sum_abs += np.log(tmpsum)
        
        return nsr_mean_sum_abs

    def compute(self, model, data_loader=None, resolution=32, batch_size=16, gpu=None, loss_func=None, repeat_times=32):
        """
        Compute the ZICO proxy score.
        Args:
            model (nn.Module): The neural network model to evaluate.
            data_loader: Placeholder, used for training loop but not needed in this context.
            resolution (int): Resolution of the input image.
            batch_size (int): Batch size for computation.
            gpu (int or str): GPU to use for computation.
            loss_func (function): Loss function to use for backpropagation.
            repeat_times (int): Number of iterations for computing gradients and averaging.
        Returns:
            float: The ZICO proxy score.
        """
        device = torch.device(f"cuda:{gpu}" if gpu is not None and torch.cuda.is_available() else "cpu")
        model = model.to(device)

        grad_dict = {}
        model.train()

        # Training loop to compute gradients
        for i, batch in enumerate(data_loader):
            model.zero_grad()
            data, label = batch[0], batch[1]
            data, label = data.to(device), label.to(device)

            logits = model(data)
            loss = loss_func(logits, label)
            loss.backward()
            grad_dict = self.getgrad(model, grad_dict, i)

        # Calculate ZICO score based on gradients
        zico_score = self.calculate_zico(grad_dict)
        return zico_score

if __name__ == "__main__":
    import argparse

    def parse_cmd_options(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size for ZICO computation.')
        parser.add_argument('--input_image_size', type=int, default=32, help='Input image resolution.')
        parser.add_argument('--repeat_times', type=int, default=32, help='Number of repeats for averaging.')
        parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use.')
        return parser.parse_args(argv)

    args = parse_cmd_options(sys.argv[1:])
    hparams = parse_configuration(sys.argv[1])  # Assuming configuration is passed as the first argument
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ImageClassification(hparams=hparams).modules["classifier"]

    # ZicoProxy initialization
    proxy = ZicoProxy()

    # Assuming loss function is provided, typically CrossEntropyLoss for classification tasks
    loss_func = nn.CrossEntropyLoss()

    # Compute ZICO score
    start_time = time.time()
    zico_score = proxy.compute(
        model=model,
        data_loader=None,  # Assuming data loader is passed during training, but can be None for now
        resolution=args.input_image_size,
        batch_size=args.batch_size,
        gpu=args.gpu,
        loss_func=loss_func,
        repeat_times=args.repeat_times
    )
    time_cost = (time.time() - start_time) / args.repeat_times

    print(f'ZICO Score={zico_score:.4g}, Time Cost={time_cost:.4g} seconds')
