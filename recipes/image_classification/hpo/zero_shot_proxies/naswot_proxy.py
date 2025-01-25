import os
import sys
import time
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .base_proxy import BaseProxy  # Import the abstract base class
from micromind.networks.phinet import ReLUMax

class NASWOTProxy(BaseProxy):
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
    def logdet(K):
        """
        Compute the log determinant of a matrix.
        """
        s, ld = np.linalg.slogdet(K)
        return ld

    @staticmethod
    def get_batch_jacobian(net, x):
        """
        Compute the Jacobian of the network output with respect to the input.
        """
        net.zero_grad()
        x.requires_grad_(True)
        y = net(x)
        y.backward(torch.ones_like(y))
        jacob = x.grad.detach()
        return jacob, y.detach()

    def compute(self, model, data_loader=None, resolution=32, batch_size=16, gpu=None):
        """
        Compute the NASWOT proxy score.
        Args:
            model (nn.Module): The neural network model to evaluate.
            data_loader: Placeholder, not used in NASWOT but part of the BaseProxy interface.
            resolution (int): Resolution of the input image.
            batch_size (int): Batch size for NASWOT computation.
            gpu (int or str): GPU to use for computation.
        Returns:
            float: The NASWOT proxy score.
        """
        device = torch.device(f"cuda:{gpu}" if gpu is not None and torch.cuda.is_available() else "cpu")
        model = model.to(device)

        torch.manual_seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.network_weight_gaussian_init(model)
        input_tensor = torch.randn(size=[batch_size, 3, resolution, resolution]).to(device)
        model.K = np.zeros((batch_size, batch_size))

        def counting_forward_hook(module, inp, out):
            try:
                if not module.visited_backwards:
                    return
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)
                x = (inp > 0).float()
                K = x @ x.t()
                K2 = (1. - x) @ (1. - x.t())
                model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()
            except Exception as err:
                print('Error in counting_forward_hook:')
                raise err

        def counting_backward_hook(module, inp, out):
            module.visited_backwards = True

        # Register hooks for activation tracking
        for _, module in model.named_modules():
            if isinstance(module, (ReLUMax, nn.ReLU, nn.Hardswish)):
                module.visited_backwards = True
                module.register_forward_hook(counting_forward_hook)
                module.register_backward_hook(counting_backward_hook)

        # Forward pass to compute the NASWOT score
        input_tensor = input_tensor.to(device)
        jacobs, y = self.get_batch_jacobian(model, input_tensor)

        # Compute score using the determinant of K
        score = self.logdet(model.K)
        return float(score)

if __name__ == "__main__":
    import argparse
    from micromind.utils import parse_configuration
    from recipes.image_classification.train import ImageClassification

    def parse_cmd_options(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size for NASWOT computation.')
        parser.add_argument('--input_image_size', type=int, default=32, help='Input image resolution.')
        parser.add_argument('--repeat_times', type=int, default=32, help='Number of repeats for averaging.')
        parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use.')
        return parser.parse_args(argv)

    args = parse_cmd_options(sys.argv[1:])
    hparams = parse_configuration(sys.argv[1])  # Assuming configuration is passed as the first argument
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ImageClassification(hparams=hparams).modules["classifier"]

    # NASWOTProxy initialization
    proxy = NASWOTProxy()

    # Compute NASWOT scores
    start_time = time.time()
    scores = [
        proxy.compute(model=model, resolution=args.input_image_size, batch_size=args.batch_size, gpu=args.gpu)
        for _ in range(args.repeat_times)
    ]
    avg_score = sum(scores) / args.repeat_times
    time_cost = (time.time() - start_time) / args.repeat_times

    print(f'Average NASWOT Score={avg_score:.4g}, Time Cost={time_cost:.4g} seconds')
