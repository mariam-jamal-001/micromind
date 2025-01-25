import optuna

class HyperparameterSampler:
    """
    A class to sample hyperparameters using different Optuna samplers.
    """

    def __init__(self, sampler_type="random"):
        """
        Initialize the sampler.

        Args:
            sampler_type (str): Sampling algorithm to use ('random', 'tpe', etc.).
        """
        self.sampler = self._get_sampler(sampler_type)
        self.study = optuna.create_study(direction='minimize', sampler=self.sampler)

    def _get_sampler(self, sampler_type):
        """
        Get the appropriate Optuna sampler.

        Args:
            sampler_type (str): Type of sampler ('random', 'tpe', etc.).

        Returns:
            optuna.samplers.BaseSampler: An instance of the chosen sampler.
        """
        if sampler_type == "random":
            return optuna.samplers.RandomSampler()
        elif sampler_type == "tpe":
            return optuna.samplers.TPESampler()
        else:
            raise ValueError(f"Unsupported sampler type: {sampler_type}")

    def sample(self, trial):
        """
        Sample hyperparameters from the search space.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.

        Returns:
            dict: Sampled hyperparameters.
        """
        alpha = trial.suggest_float("alpha", 0.1, 3)
        beta = trial.suggest_float("beta", 0.1, 1)
        t_zero = trial.suggest_int("t_zero", 3, 6)
        num_layers = trial.suggest_int("num_layers", 1, 10)
        return {"alpha": alpha, "beta": beta, "t_zero": t_zero, "num_layers": num_layers}
