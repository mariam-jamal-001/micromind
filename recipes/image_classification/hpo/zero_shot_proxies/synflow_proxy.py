import os
import sys
import time
import torch
from torch import nn
import numpy as np
from .base_proxy import BaseProxy  # Import the abstract base class
from micromind.utils import parse_configuration
from recipes.image_classification.train import ImageClassification

class SynFlowProxy(BaseProxy):
    @staticmethod
    def network_weight_gaussian_init(net: nn.Module):
        """
        Initialize network weights using Gaussian distribution.
        """
        with torch.no_grad():
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
        return net

    @staticmethod
    def get_layer_metric_array(net, metric, mode):
        """
        Get the metric for each layer in the network.
        """
        metric_array = []
        for layer in net.modules():
            if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
                continue
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                metric_array.append(metric(layer))
        return metric_array

    @staticmethod
    def compute_synflow_per_weight(net, inputs, mode):
        """
        Compute SynFlow for each weight in the network.
        """
        device = inputs.device

        @torch.no_grad()
        def linearize(net):
            """
            Linearize the network parameters (absolute values) and keep track of their signs.
            """
            signs = {}
            for name, param in net.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(net, signs):
            """
            Restore the original parameter values using the saved signs.
            """
            for name, param in net.state_dict().items():
                if 'weight_mask' not in name:
                    param.mul_(signs[name])

        # Linearize parameters and keep track of signs
        signs = linearize(net)

        # Compute gradients
        net.zero_grad()
        net.double()
        output = net.forward(inputs.double())
        torch.sum(output).backward()

        # SynFlow metric computation
        def synflow(layer):
            if layer.weight.grad is not None:
                return torch.abs(layer.weight * layer.weight.grad)
            return torch.zeros_like(layer.weight)

        grads_abs = SynFlowProxy.get_layer_metric_array(net, synflow, mode)

        # Restore the original parameter values
        nonlinearize(net, signs)

        return grads_abs

    def compute(self, model, data_loader=None, resolution=32, batch_size=16, gpu=None):
        """
        Compute the SynFlow proxy score.
        Args:
            model (nn.Module): The neural network model to evaluate.
            data_loader: Placeholder, not used in SynFlow but part of the BaseProxy interface.
            resolution (int): Resolution of the input image.
            batch_size (int): Batch size for SynFlow computation.
            gpu (int or str): GPU to use for computation.
        Returns:
            float: The SynFlow proxy score.
        """
        device = torch.device(f"cuda:{gpu}" if gpu is not None and torch.cuda.is_available() else "cpu")
        model = model.to(device)

        self.network_weight_gaussian_init(model)
        input_tensor = torch.randn(size=[batch_size, 3, resolution, resolution]).to(device)

        grads_abs_list = self.compute_synflow_per_weight(net=model, inputs=input_tensor, mode='')
        score = 0
        for grad_abs in grads_abs_list:
            if len(grad_abs.shape) == 4:
                score += float(torch.mean(torch.sum(grad_abs, dim=[1, 2, 3])))
            elif len(grad_abs.shape) == 2:
                score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
            else:
                raise RuntimeError("Unexpected tensor shape!")

        return -1 * score

if __name__ == "__main__":
    import argparse

    def parse_cmd_options(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size for SynFlow computation.')
        parser.add_argument('--input_image_size', type=int, default=32, help='Input image resolution.')
        parser.add_argument('--repeat_times', type=int, default=32, help='Number of repeats for averaging.')
        parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use.')
        return parser.parse_args(argv)

    args = parse_cmd_options(sys.argv[1:])
    hparams = parse_configuration(sys.argv[1])  # Assuming configuration is passed as the first argument
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ImageClassification(hparams=hparams).modules["classifier"]

    # SynFlowProxy initialization
    proxy = SynFlowProxy()

    # Compute SynFlow scores
    start_time = time.time()
    scores = [
        proxy.compute(model=model, resolution=args.input_image_size, batch_size=args.batch_size, gpu=args.gpu)
        for _ in range(args.repeat_times)
    ]
    avg_score = sum(scores) / args.repeat_times
    time_cost = (time.time() - start_time) / args.repeat_times

    print(f'Average SynFlow Score={avg_score:.4g}, Time Cost={time_cost:.4g} seconds')
