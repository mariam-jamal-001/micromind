import optuna
import torch
import torch.nn as nn
from prepare_data import create_loaders, setup_mixup
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

def objective(trial):
    """
    Objective function to be used by Optuna for optimization.
    """
    hparams = parse_configuration("config.yaml")  # Load your config file

    # Define search space for hyperparameters using Optuna's trial.suggest methods
    hparams.alpha = trial.suggest_float("alpha", 0.3, 3.0)
    hparams.beta = trial.suggest_float("beta", 0.5, 1.0)
    hparams.t_zero = trial.suggest_int("t_zero", 3, 6)
    
    # Set up data loaders
    loaders = create_loaders(hparams)
    
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

    # Create an instance of the model
    mind = ImageClassification(hparams=hparams)
    top1 = mm.Metric("top1_acc", top_k_accuracy(k=1), eval_only=True)
    top5 = mm.Metric("top5_acc", top_k_accuracy(k=5), eval_only=True)

    # Start training
    mind.train(
        epochs=hparams.epochs,
        datasets={"train": loaders[0], "val": loaders[1]},
        metrics=[top5, top1],
        checkpointer=None,
        verbose=True
    )

    # Testing the model after training
    test_results = mind.test(
        datasets={"test": loaders[1]},
        metrics=[top5, top1],
        verbose=True
    )
    
    # Return the test loss as the objective to minimize
    return test_results["test_loss"]

def run_optuna_optimization():
    # Create an Optuna study to optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    # Print best hyperparameters and objective value
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    run_optuna_optimization()
