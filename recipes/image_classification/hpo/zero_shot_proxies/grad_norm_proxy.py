import os
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
from .base_proxy import BaseProxy  # Import the abstract base class

class GradNormProxy(BaseProxy):
    def network_weight_gaussian_init(self, net: nn.Module):
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

    def cross_entropy(self, logit, target):
        """
        Custom cross-entropy loss implementation for one-hot targets.
        """
        prob_logit = F.log_softmax(logit, dim=1)
        loss = -(target * prob_logit).sum(dim=1).mean()
        return loss

    def compute(self, model, data_loader=None, resolution=32, batch_size=16, gpu=None):
        """
        Compute the GradNorm proxy score.
        Args:
            model (nn.Module): The neural network model to evaluate.
            data_loader: Placeholder, not used in GradNorm but part of the BaseProxy interface.
            resolution (int): Resolution of the input image.
            batch_size (int): Batch size for GradNorm computation.
            gpu (int or str): GPU to use for computation.
        Returns:
            float: The GradNorm proxy score.
        """
        model.train()
        model.requires_grad_(True)
        model.zero_grad()

        # Initialize model weights
        self.network_weight_gaussian_init(model)

        # Generate random input and forward pass
        device = torch.device(f"cuda:{gpu}" if gpu is not None and torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_tensor = torch.randn(size=[batch_size, 3, resolution, resolution]).to(device)
        output = model(input_tensor)

        # Generate random one-hot targets
        num_classes = output.shape[1]
        y = torch.randint(low=0, high=num_classes, size=[batch_size]).to(device)
        one_hot_y = F.one_hot(y, num_classes).float()

        # Compute loss and gradients
        loss = self.cross_entropy(output, one_hot_y)
        loss.backward()

        # Compute the GradNorm score
        norm2_sum = 0
        with torch.no_grad():
            for p in model.parameters():
                if hasattr(p, 'grad') and p.grad is not None:
                    norm2_sum += torch.norm(p.grad) ** 2
        grad_norm = float(torch.sqrt(norm2_sum))

        return grad_norm

if __name__ == "__main__":
    # Command-line argument parsing
    import argparse
    from micromind.utils import parse_configuration
    from recipes.image_classification.train import ImageClassification

    def parse_cmd_options(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size for GradNorm computation.')
        parser.add_argument('--input_image_size', type=int, default=32, help='Input image resolution.')
        parser.add_argument('--repeat_times', type=int, default=32, help='Number of repeats for averaging.')
        parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use.')
        return parser.parse_args(argv)

    args = parse_cmd_options(sys.argv[1:])
    hparams = parse_configuration(sys.argv[1])  # Assuming configuration is passed as the first argument
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ImageClassification(hparams=hparams).modules["classifier"]

    # GradNormProxy initialization
    proxy = GradNormProxy()

    # Compute GradNorm scores
    start_time = time.time()
    scores = [
        proxy.compute(model=model, resolution=args.input_image_size, batch_size=args.batch_size, gpu=args.gpu)
        for _ in range(args.repeat_times)
    ]
    avg_score = sum(scores) / args.repeat_times
    time_cost = (time.time() - start_time) / args.repeat_times

    print(f'Average GradNorm Score={avg_score:.4g}, Time Cost={time_cost:.4g} seconds')
