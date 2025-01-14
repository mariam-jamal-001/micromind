"""
This code runs the image classification training loop. It tries to support as much
as timm's functionalities as possible.

For compatibility the prefetcher, re_split and JSDLoss are disabled.

To run the training script, use this command:
    python train.py cfg/phinet.py

You can change the configuration or override the parameters as you see fit.

Authors:
    - Francesco Paissan, 2023
"""

import torch
import torch.nn as nn
from .prepare_data import create_loaders, setup_mixup
from timm.loss import (
    BinaryCrossEntropy,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)

import micromind as mm
from typing import Tuple, List
from micromind.networks import PhiNet, XiNet
from micromind.utils import parse_configuration
from argparse import Namespace
from functools import partial
import sys
import os


class ImageClassification(mm.MicroMind):
    """Implements an image classification class. Provides support
    for timm augmentation and loss functions."""

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)

        if hparams.model == "phinet":
            self.modules["classifier"] = PhiNet(
                input_shape=hparams.input_shape,
                alpha=hparams.alpha,
                num_layers=hparams.num_layers,
                beta=hparams.beta,
                t_zero=hparams.t_zero,
                compatibility=False,
                divisor=hparams.divisor,
                downsampling_layers=hparams.downsampling_layers,
                return_layers=hparams.return_layers,
                # classification-specific
                include_top=True,
                num_classes=hparams.num_classes,
                h_swish=hparams.h_swish
            )
        elif hparams.model == "xinet":
            self.modules["classifier"] = XiNet(
                input_shape=hparams.input_shape,
                alpha=hparams.alpha,
                gamma=hparams.gamma,
                num_layers=hparams.num_layers,
                return_layers=hparams.return_layers,
                # classification-specific
                include_top=True,
                num_classes=hparams.num_classes,
            )

        self.mixup_fn, _ = setup_mixup(hparams)

        print(f"Number of parameters for each module: {self.compute_params()}")
        print(
            f"Number of MAC for each module: {self.compute_macs(hparams.input_shape)}"
        )

    def setup_criterion(self):
        """Setup of the loss function based on augmentation strategy."""
        # setup loss function
        if (
            self.hparams.mixup > 0
            or self.hparams.cutmix > 0.0
            or self.hparams.cutmix_minmax is not None
        ):
            # smoothing is handled with mixup target transform which outputs sparse,
            # soft targets
            if self.hparams.bce_loss:
                train_loss_fn = BinaryCrossEntropy(
                    target_threshold=self.hparams.bce_target_thresh
                )
            else:
                train_loss_fn = SoftTargetCrossEntropy()
        elif self.hparams.smoothing:
            if self.hparams.bce_loss:
                train_loss_fn = BinaryCrossEntropy(
                    smoothing=self.hparams.smoothing,
                    target_threshold=self.hparams.bce_target_thresh,
                )
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(
                    smoothing=self.hparams.smoothing
                )
        else:
            train_loss_fn = nn.CrossEntropyLoss()

        return train_loss_fn

    def forward(self, batch):
        """Computes forward step for image classifier.

        Arguments
        ---------
        batch : List[torch.Tensor, torch.Tensor]
            Batch containing the images and labels.

        Returns
        -------
        Predicted class and augmented class. : Tuple[torch.Tensor, torch.Tensor]
        """
        img, target = batch
        if not self.hparams.prefetcher:
            img, target = img.to(self.device), target.to(self.device)
            if self.mixup_fn is not None:
                img, target = self.mixup_fn(img, target)

        return (self.modules["classifier"](img), target)

    def compute_loss(self, pred, batch):
        """Sets up the loss function and computes the criterion.

        Arguments
        ---------
        pred : Tuple[torch.Tensor, torch.Tensor]
            Predicted class and augmented class.
        batch : List[torch.Tensor, torch.Tensor]
            Same batch as input to the forward step.

        Returns
        -------
        Cost function. : torch.Tensor
        """
        self.criterion = self.setup_criterion()

        # taking it from pred because it might be augmented
        return self.criterion(pred[0], pred[1])

    def configure_optimizers(self):
        """Configures the optimizes and, eventually the learning rate scheduler."""
        opt = torch.optim.Adam(self.modules.parameters(), lr=3e-4, weight_decay=0.0005)
        return opt


def top_k_accuracy(k=1):
    """
    Computes the top-K accuracy.

    Arguments
    ---------
    k : int
       Number of top elements to consider for accuracy.

    Returns
    -------
        accuracy : Callable
            Top-K accuracy.
    """

    def acc(pred, batch):
        if pred[1].ndim == 2:
            target = pred[1].argmax(1)
        else:
            target = pred[1]
        _, indices = torch.topk(pred[0], k, dim=1)
        correct = torch.sum(indices == target.view(-1, 1))
        accuracy = correct.item() / target.size(0)

        return torch.Tensor([accuracy]).to(pred[0].device)

    return acc


def run_one_experiment(hpo: Tuple, hparams: Namespace, loaders: List):
    """This runs a training for a specific configuration.
    It is wrapped in this function for compatibility with HPO pipelines.

    Arguments
    ---------
    hpo : Optional[Tuple]
        Passed only when this serves as evaluation function for HPO, ignored
        otherwise.

    Returns
    -------
    Objective function for HPO. : float
    """
    exp_configuration = ""
    if hpo is not None:
        # HPO is running
        print("HPO proposed the following configuration: ")
        print(hpo)
        for conf in hpo:
            # loops through all suggested parameters
            setattr(hparams, conf, hpo[conf])

        exp_configuration = "_".join([f"{a}_{hpo[a]:.2f}" for a in hpo])

    exp_folder = mm.utils.checkpointer.create_experiment_folder(
        hparams.output_folder, hparams.experiment_name + "+" + exp_configuration
    )

    checkpointer = mm.utils.checkpointer.Checkpointer(
        exp_folder,
        hparams=hparams if hpo is None else None,
        key="loss",
        verbose=hpo is None,
    )

    mind = ImageClassification(hparams=hparams)

    # validate complexity of configuration
    if sum(mind.compute_params().values()) > getattr(hparams, "MAX_PARAMS", 1e15):
        # requested configuration yields really big network
        # return high value so it's not selected
        return float(1e9)

    top1 = mm.Metric("top1_acc", top_k_accuracy(k=1), eval_only=True)
    top5 = mm.Metric("top5_acc", top_k_accuracy(k=5), eval_only=True)

    mind.train(
        epochs=hparams.epochs,
        datasets={"train": loaders[0], "val": loaders[1]},
        metrics=[top5, top1],
        checkpointer=checkpointer,
        verbose=hpo is None,
        debug=hparams.debug,
    )

    test_results = mind.test(
        datasets={"test": loaders[1]}, metrics=[top1, top5], verbose=hpo is None
    )

    return test_results["test_loss"]


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please pass the configuration file to the script."

    # get experiment configuration
    hparams = parse_configuration(sys.argv[1])

    loaders = create_loaders(hparams)

    if hasattr(hparams, "search_space"):
        from hyperopt import fmin, tpe, Trials

        trials = Trials()
        obj = partial(run_one_experiment, hparams=hparams, loaders=loaders)
        best = fmin(
            fn=obj,
            space=hparams.search_space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=hparams.hpo_trials,
        )

        import pandas as pd

        trials_df = pd.DataFrame([pd.Series(t["misc"]["vals"]) for t in trials])
        trials_df["loss"] = [t["result"]["loss"] for t in trials]
        trials_df.to_csv(
            os.path.join(hparams.output_folder, f"{hparams.experiment_name}.csv")
        )

    else:
        # wrapped in here for HPO
        run_one_experiment(None, hparams, loaders)
