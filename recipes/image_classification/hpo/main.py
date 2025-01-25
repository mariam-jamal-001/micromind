import optuna
from optuna import Trial

def define_search_space(trial: Trial):
    # alpha = 3
    alpha = trial.suggest_float("alpha", 0.1, 10)  # alpha: [0, 1]
    # beta = 1
    beta = trial.suggest_float("beta", 0.5, 5)   # beta: [0, 10]
    # t_zero = 5
    t_zero = trial.suggest_int("t_zero", 1, 10)       # t_zero: [0, 100]
    #num_layers = 7
    num_layers = trial.suggest_int("num_layers", 1, 10)
    
    return alpha, beta, t_zero, num_layers

