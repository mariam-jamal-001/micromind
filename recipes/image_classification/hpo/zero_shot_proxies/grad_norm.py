import os, sys, time
sys.path.append('/home/majam001/thesis_mariam/micromind/recipes/image_classification/')
import torch
from torch import nn
import numpy as np
from recipes.image_classification.train import ImageClassification
from micromind.utils import parse_configuration
import argparse

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

import torch.nn.functional as F
def cross_entropy(logit, target):
    # target must be one-hot format!!
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss

def compute_gradnorm_score(gpu, model, resolution, batch_size):

    model.train()
    model.requires_grad_(True)

    model.zero_grad()

    # if gpu is not None:
    #     torch.cuda.set_device(gpu)
    #     model = model.cuda(gpu)

    network_weight_gaussian_init(model)
    input = torch.randn(size=[batch_size, 3, resolution, resolution])
    input = input.to(gpu)
    output = model(input)
    # y_true = torch.rand(size=[batch_size, output.shape[1]], device=torch.device('cuda:{}'.format(gpu))) + 1e-10
    # y_true = y_true / torch.sum(y_true, dim=1, keepdim=True)

    num_classes = output.shape[1]
    y = torch.randint(low=0, high=num_classes, size=[batch_size])

    one_hot_y = F.one_hot(y, num_classes).float()
    if gpu is not None:
        one_hot_y = one_hot_y.cuda(gpu)

    loss = cross_entropy(output, one_hot_y)
    loss.backward()
    norm2_sum = 0
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                norm2_sum += torch.norm(p.grad) ** 2

    grad_norm = float(torch.sqrt(norm2_sum))

    return grad_norm

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
        the_score = compute_gradnorm_score(gpu=device, model=the_model,
                            resolution=args.input_image_size, batch_size=args.batch_size)

    time_cost = (time.time() - start_timer) / args.repeat_times

    print(f'Grad Norm={the_score:.4g}, time cost={time_cost:.4g} second(s)')