import os
import sys
import time
import torch
import numpy as np
from micromind.utils import parse_configuration
from recipes.image_classification.train import ImageClassification
from micromind.networks.phinet import ReLUMax
from torch import nn
import argparse

# Helper function for computing NTK score
def compute_NTK_score(gpu, model, resolution, batch_size):
    device = torch.device(f"cuda:{gpu}" if gpu is not None else "cpu")
    model = model.to(device)
    grads = []
    for _ in range(1):  # num_batch = 1 as specified
        inputs = torch.randn((batch_size, 3, resolution, resolution), device=device)
        model.zero_grad()
        logit = model(inputs)
        logit.backward(torch.ones_like(logit), retain_graph=True)

        grad_list = []
        for name, W in model.named_parameters():
            if 'weight' in name and W.grad is not None:
                grad_list.append(W.grad.view(-1).detach())
        grads.append(torch.cat(grad_list, -1))

    grads = torch.stack(grads, 0)
    ntk = torch.einsum('nc,mc->nm', grads, grads)
    eigenvalues, _ = torch.linalg.eigh(ntk)
    cond = (eigenvalues[-1] / eigenvalues[0]).item()
    return -cond  # Negative NTK score

# Helper function for computing RN score
def compute_RN_score(model, batch_size, image_size, num_batch, gpu):
    device = torch.device(f"cuda:{gpu}" if gpu is not None else "cpu")
    model = model.to(device)
    lrc_model = Linear_Region_Collector(models=[model], input_size=(batch_size, 3, image_size, image_size),
                                        gpu=gpu, sample_batch=num_batch)
    return lrc_model.forward_batch_sample()[0]

class LinearRegionCount(object):
    """Computes and stores the average and current value of linear regions."""
    def __init__(self, n_samples, gpu=None):
        self.ActPattern = {}
        self.n_LR = -1
        self.n_samples = n_samples
        self.ptr = 0
        self.activations = None
        self.device = gpu

    @torch.no_grad()
    def update2D(self, activations):
        n_batch = activations.size()[0]
        n_neuron = activations.size()[1]
        self.n_neuron = n_neuron
        if self.activations is None:
            self.activations = torch.zeros(self.n_samples, n_neuron)
            self.activations = self.activations.to(self.device)
        self.activations[self.ptr:self.ptr + n_batch] = torch.sign(activations)  # After ReLU
        self.ptr += n_batch

    @torch.no_grad()
    def calc_LR(self):
        res = torch.matmul(self.activations.half(), (1 - self.activations).T.half())
        res += res.T.clone()
        res = 1 - torch.sign(res)
        res = res.sum(1)
        res = 1. / res.float()
        self.n_LR = res.sum().item()
        del self.activations, res
        self.activations = None
        if self.device is not None:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def update1D(self, activationList):
        code_string = ''
        for key, value in activationList.items():
            n_neuron = value.size()[0]
            for i in range(n_neuron):
                code_string += '1' if value[i] > 0 else '0'
        if code_string not in self.ActPattern:
            self.ActPattern[code_string] = 1

    def getLinearReginCount(self):
        if self.n_LR == -1:
            self.calc_LR()
        return self.n_LR


class Linear_Region_Collector:
    def __init__(self, models=[], input_size=(64, 3, 32, 32), gpu=None, sample_batch=1, dataset=None, data_path=None, seed=0):
        self.models = models
        self.input_size = input_size
        self.sample_batch = sample_batch
        self.dataset = dataset
        self.data_path = data_path
        self.seed = seed
        self.device = gpu
        self.reinit(models, input_size, sample_batch, seed)

    def reinit(self, models=None, input_size=None, sample_batch=None, seed=None):
        if models is not None:
            self.models = models
            for model in self.models:
                self.register_hook(model)
            self.LRCounts = [LinearRegionCount(self.input_size[0] * self.sample_batch, gpu=self.device) for _ in range(len(models))]
        if input_size is not None or sample_batch is not None:
            self.input_size = input_size
            if sample_batch is not None:
                self.sample_batch = sample_batch
        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            if self.device is not None:
                torch.cuda.manual_seed(seed)
        self.clear()

    def clear(self):
        self.LRCounts = [LinearRegionCount(self.input_size[0] * self.sample_batch) for _ in range(len(self.models))]
        if self.device is not None:
            torch.cuda.empty_cache()

    def register_hook(self, model):
        for m in model.modules():
            if isinstance(m, (nn.Hardswish, ReLUMax, nn.ReLU)):
                m.register_forward_hook(self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())  # For ReLU

    def forward_batch_sample(self):
        for _ in range(self.sample_batch):
            inputs = torch.randn(self.input_size, device=self.device)
            for model, LRCount in zip(self.models, self.LRCounts):
                self.forward(model, LRCount, inputs)
        return [LRCount.getLinearReginCount() for LRCount in self.LRCounts]

    def forward(self, model, LRCount, input_data):
        self.interFeature = []
        with torch.no_grad():
            model.forward(input_data)
            if len(self.interFeature) == 0: return
            feature_data = torch.cat([f.view(input_data.size(0), -1) for f in self.interFeature], 1)
            LRCount.update2D(feature_data)


class TE_NASProxy:
    def __init__(self):
        pass

    def compute_ntk_score(self, model, batch_size, resolution, gpu):
        return compute_NTK_score(gpu=gpu, model=model, resolution=resolution, batch_size=batch_size)

    def compute_rn_score(self, model, batch_size, resolution, gpu):
        return compute_RN_score(model=model, batch_size=batch_size, image_size=resolution, num_batch=1, gpu=gpu)

    def compute(self, model, resolution=32, batch_size=16, gpu=None):
        device = torch.device(f"cuda:{gpu}" if gpu is not None and torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        ntk_score = self.compute_ntk_score(model=model, batch_size=batch_size, resolution=resolution, gpu=gpu)
        rn_score = self.compute_rn_score(model=model, batch_size=batch_size, resolution=resolution, gpu=gpu)

        return ntk_score + rn_score


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for TE_NAS computation.')
    parser.add_argument('--input_image_size', type=int, default=32, help='Input image resolution.')
    parser.add_argument('--repeat_times', type=int, default=32, help='Number of repeats for averaging.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use.')
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_cmd_options(sys.argv[1:])
    hparams = parse_configuration(sys.argv[1])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = ImageClassification(hparams=hparams).modules["classifier"]

    proxy = TE_NASProxy()

    start_time = time.time()
    scores = [
        proxy.compute(model=model, resolution=args.input_image_size, batch_size=args.batch_size, gpu=args.gpu)
        for _ in range(args.repeat_times)
    ]
    avg_score = sum(scores) / args.repeat_times
    time_cost = (time.time() - start_time) / args.repeat_times

    print(f'Average TE_NAS Score={avg_score:.4g}, Time Cost={time_cost:.4g} seconds')
