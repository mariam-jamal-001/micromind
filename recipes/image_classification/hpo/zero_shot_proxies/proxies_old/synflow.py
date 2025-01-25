'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from
https://github.com/SamsungLabs/zero-cost-nas
'''


# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


import os, sys, time
sys.path.append('/home/majam001/thesis_mariam/micromind/recipes/image_classification/')
import torch
from torch import nn
import numpy as np
from recipes.image_classification.train import ImageClassification
from micromind.utils import parse_configuration
import argparse

import torch

def network_weight_gaussian_init(net: nn.Module):
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
            else:
                continue

    return net

def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))

    return metric_array



def compute_synflow_per_weight(net, inputs, mode):
    device = inputs.device

    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)

    # Compute gradients with input of 1s
    net.zero_grad()
    net.double()
    #input_dim = list(inputs[0, :].shape)
    #inputs = torch.ones([1] + input_dim).double().to(device)
    output = net.forward(inputs.double())
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow, mode)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs

def compute_synflow_score(gpu, model, resolution, batch_size):
    model.train()
    model.requires_grad_(True)

    model.zero_grad()

    # if gpu is not None:
    #     torch.cuda.set_device(gpu)
    #     model = model.cuda(gpu)

    network_weight_gaussian_init(model)
    input = torch.randn(size=[batch_size, 3, resolution, resolution])
    input = input.to(gpu)

    grads_abs_list = compute_synflow_per_weight(net=model, inputs=input, mode='')
    score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1,2,3])))
        elif len(grad_abs.shape) == 2:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        else:
            raise RuntimeError('!!!')


    return -1 * score

def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=32,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

if __name__ == "__main__":
    # opt = global_utils.parse_cmd_options(sys.argv)
    args = parse_cmd_options(sys.argv)
    # the_model = ModelLoader.get_model(opt, sys.argv)
    hparams = parse_configuration(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    the_model = ImageClassification(hparams=hparams).modules["classifier"].to(device)

    # if args.gpu is not None:
    #     the_model = the_model.to(args.gpu)


    start_timer = time.time()

    for repeat_count in range(args.repeat_times):
        the_score = compute_synflow_score(gpu=device, model=the_model,
                            resolution=args.input_image_size, batch_size=args.batch_size)

    time_cost = (time.time() - start_timer) / args.repeat_times

    print(f'Syn Flow={the_score:.4g}, time cost={time_cost:.4g} second(s)')