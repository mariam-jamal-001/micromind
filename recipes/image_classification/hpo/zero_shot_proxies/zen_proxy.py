import os
import sys
import time
import torch
import numpy as np
from torch import nn
from .base_proxy import BaseProxy  # Import the abstract base class
from micromind.utils import parse_configuration
from recipes.image_classification.train import ImageClassification

class ZenProxy(BaseProxy):
    def __init__(self):
        super(ZenProxy, self).__init__()

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

    def compute_zen_score(self, model, mixup_gamma, resolution, batch_size, repeat, device, fp16=False):
        """
        Compute the Zen score for the given model.
        """
        info = {}
        nas_score_list = []
        
        dtype = torch.half if fp16 else torch.float32

        with torch.no_grad():
            for repeat_count in range(repeat):
                # Initialize the weights using Gaussian distribution
                self.network_weight_gaussian_init(model)
                
                # Generate random inputs
                input_tensor = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
                input2_tensor = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
                mixup_input = input_tensor + mixup_gamma * input2_tensor
                
                # Forward pass
                output = model(input_tensor)
                mixup_output = model(mixup_input)

                # Calculate NAS score
                nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
                nas_score = torch.mean(nas_score)

                # Compute BN scaling factor
                log_bn_scaling_factor = 0.0
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                        log_bn_scaling_factor += torch.log(bn_scaling_factor)

                # Combine scores
                nas_score = torch.log(nas_score) + log_bn_scaling_factor
                nas_score_list.append(float(nas_score))

        # Calculate statistics
        std_nas_score = np.std(nas_score_list)
        avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
        avg_nas_score = np.mean(nas_score_list)

        info['avg_nas_score'] = float(avg_nas_score)
        info['std_nas_score'] = float(std_nas_score)
        info['avg_precision'] = float(avg_precision)
        
        return info

    def compute(self, model, data_loader=None, resolution=32, batch_size=16, gpu=None, mixup_gamma=1e-2, repeat_times=32, fp16=False):
        """
        Compute the Zen proxy score.
        Args:
            model (nn.Module): The neural network model to evaluate.
            data_loader: Placeholder, not used in ZenProxy but part of the BaseProxy interface.
            resolution (int): Resolution of the input image.
            batch_size (int): Batch size for computation.
            gpu (int or str): GPU to use for computation.
            mixup_gamma (float): Gamma for mixup.
            repeat_times (int): Number of repetitions for averaging.
            fp16 (bool): Whether to use FP16 precision.
        Returns:
            float: The Zen proxy score.
        """
        device = torch.device(f"cuda:{gpu}" if gpu is not None and torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Compute the Zen score
        info = self.compute_zen_score(
            model=model,
            mixup_gamma=mixup_gamma,
            resolution=resolution,
            batch_size=batch_size,
            repeat=repeat_times,
            device=device,
            fp16=fp16
        )

        return info['avg_nas_score']

if __name__ == "__main__":
    import argparse

    def parse_cmd_options(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size for Zen computation.')
        parser.add_argument('--input_image_size', type=int, default=32, help='Input image resolution.')
        parser.add_argument('--repeat_times', type=int, default=32, help='Number of repeats for averaging.')
        parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use.')
        parser.add_argument('--mixup_gamma', type=float, default=1e-2, help='Gamma for mixup.')
        return parser.parse_args(argv)

    args = parse_cmd_options(sys.argv[1:])
    hparams = parse_configuration(sys.argv[1])  # Assuming configuration is passed as the first argument
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ImageClassification(hparams=hparams).modules["classifier"]

    # ZenProxy initialization
    proxy = ZenProxy()

    # Compute Zen scores
    start_time = time.time()
    zen_score = proxy.compute(
        model=model,
        resolution=args.input_image_size,
        batch_size=args.batch_size,
        gpu=args.gpu,
        mixup_gamma=args.mixup_gamma,
        repeat_times=args.repeat_times
    )
    time_cost = (time.time() - start_time) / args.repeat_times

    print(f'Zen Score={zen_score:.4g}, Time Cost={time_cost:.4g} seconds')
