import optuna
import numpy as np
import matplotlib.pyplot as plt
import wandb  # Import for WandB integration
import pandas as pd
import torch
import logging
from micromind.networks import PhiNet

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define your hyperparameter ranges
def define_search_space(trial):
    # alpha = 3
    alpha = trial.suggest_float("alpha", 0.1, 10)  # alpha: [0, 1]
    # beta = 1
    beta = trial.suggest_float("beta", 0.5, 5)   # beta: [0, 10]
    # t_zero = 5
    t_zero = trial.suggest_int("t_zero", 1, 10)       # t_zero: [0, 100]
    #num_layers = 7
    num_layers = trial.suggest_int("num_layers", 1, 10)
    
    return alpha, beta, t_zero, num_layers

def plot_params_histogram(params_vals, bins=50):
    """
    Plot a histogram of the distribution of the number of parameters.

    Args:
        params_vals (numpy.ndarray): Array of number of parameters from the search space.
        bins (int): Number of bins to use in the histogram.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(params_vals, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Parameters')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Parameters in the Search Space')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save and show the histogram
    plt.savefig("/home/majam001/thesis_mariam/micromind/recipes/image_classification/plots/search_space/phinets_params_histogram.jpg")
    logger.info("Histogram of number of parameters saved.")


# Function to sample from the search space
def sample_and_compute(num_samples=1000):
    mac_values = []
    params_values = []
    alpha_values = []
    beta_values = []
    t_zero_values = []
    num_layers_values = []
    
    # Create an Optuna study but do not optimize (we only want to sample)
    study = optuna.create_study(direction='minimize')  # We are not minimizing anything here
    logger.info("Starting the hyperparameter sampling process...")
    
    # Sample from the search space
    for i in range(num_samples):
        logger.info(f"Sampling trial {i+1}/{num_samples}")
        trial = study.ask()  # Ask for a new sample
        alpha, beta, t_zero, num_layers = define_search_space(trial)

        logger.info(f"Trial {i+1}: alpha={alpha}, beta={beta}, t_zero={t_zero}, num_layers={num_layers}")
        
        try:
            # Create model with sampled hyperparameters
            model = PhiNet(
                    input_shape=(3, 32, 32),
                    alpha=alpha,
                    num_layers=num_layers,
                    beta=beta,
                    t_zero=t_zero,
                    compatibility=False,
                    divisor=8,
                    downsampling_layers=[5, 7],
                    return_layers=None,
                    # classification-specific
                    include_top=True,
                    num_classes=10,  # no of classes= 10 for CIFAR10
                )
            
            # Move model to GPU (if available)
            model.to(device)

            # Compute number of parameters and MACs using the inbuilt function
            mac = model.get_complexity()["MAC"]/1e6
            params = model.get_complexity()["params"]/1e6
            
            # Save the results
            params_values.append(params)
            mac_values.append(mac)
            alpha_values.append(alpha)
            beta_values.append(beta)
            t_zero_values.append(t_zero)
            num_layers_values.append(num_layers)

            # Log individual trials to WandB
            wandb.log({
                "alpha": alpha,
                "beta": beta,
                "t_zero": t_zero,
                "num_layers": num_layers,
                "params": params,
                "mac": mac
            })
            
            logger.info(f"params={params}, mac={mac}")
        
        except Exception as e:
            logger.error(f"Error in trial {i+1}: {e}")
        
        # Clear GPU memory after the trial
        del model  # Delete model to free memory
        torch.cuda.empty_cache()  # Clear any unused memory in the GPU
    
    logger.info("Hyperparameter sampling process completed.")
    return np.array(params_values), np.array(mac_values), np.array(alpha_values), np.array(beta_values), np.array(t_zero_values), np.array(num_layers_values)

# Visualize the results
def plot_results(alpha_vals, beta_vals, t_zero_vals, num_layers_vals, params_vals, mac_vals):
    plt.figure(figsize=(20, 6))
    

    # alpha vs num_params
    plt.subplot(2, 4, 1)
    plt.scatter(alpha_vals, params_vals, alpha=0.6)
    plt.xlabel('Alpha')
    plt.ylabel('Number of Parameters (M)')
    plt.title('Alpha vs Number of Parameters (M)')
    plt.grid()

    # beta vs num_params
    plt.subplot(2, 4, 2)
    plt.scatter(beta_vals, params_vals, alpha=0.6)
    plt.xlabel('Beta')
    plt.ylabel('Number of Parameters (M)')
    plt.title('Beta vs Number of Parameters (M)')
    plt.grid()

    # t_zero vs num_params
    plt.subplot(2, 4, 3)
    plt.scatter(t_zero_vals, params_vals, alpha=0.6)
    plt.xlabel('T_zero')
    plt.ylabel('Number of Parameters (M)')
    plt.title('T_zero vs Number of Parameters (M)')
    plt.grid()

    # num_layers vs num_params
    plt.subplot(2, 4, 4)
    plt.scatter(num_layers_vals, params_vals, alpha=0.6)
    plt.xlabel('Num Layers')
    plt.ylabel('Number of Parameters (M)')
    plt.title('Num Layers vs Number of Parameters (M)')
    plt.grid()

    # alpha vs mac
    plt.subplot(2, 4, 5)
    plt.scatter(alpha_vals, mac_vals, alpha=0.6)
    plt.xlabel('Alpha')
    plt.ylabel('MACs (M)')
    plt.title('Alpha vs MACs (M) ')
    plt.grid()

    # beta vs mac
    plt.subplot(2, 4, 6)
    plt.scatter(beta_vals, mac_vals, alpha=0.6)
    plt.xlabel('Beta')
    plt.ylabel('MACs (M)')
    plt.title('Beta vs MACs (M)')
    plt.grid()

    # t_zero vs mac
    plt.subplot(2, 4, 7)
    plt.scatter(t_zero_vals, mac_vals, alpha=0.6)
    plt.xlabel('T_zero')
    plt.ylabel('MACs (M)')
    plt.title('T_zero vs MACs (M)')
    plt.grid()

    # num_layers vs mac
    plt.subplot(2, 4, 8)
    plt.scatter(num_layers_vals, mac_vals, alpha=0.6)
    plt.xlabel('Num Layers')
    plt.ylabel('MACs (M)')
    plt.title('Num Layers vs MACs (M)')
    plt.grid()

    plt.tight_layout()
    plt.savefig("/home/majam001/thesis_mariam/micromind/recipes/image_classification/plots/search_space/phinets_analyze_2.jpg")
    logger.info("Results plotted and saved.")
    plt.show()

# Save results to a CSV and log to WandB
def save_results_to_csv_and_wandb(alpha_vals, beta_vals, t_zero_vals, num_layers_vals, params_vals, mac_vals):
    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'alpha': alpha_vals,
        'beta': beta_vals,
        't_zero': t_zero_vals,
        'num_layers': num_layers_vals,
        'params': params_vals,
        'mac': mac_vals
    })

    # Save the DataFrame to a CSV file
    results_df.to_csv('hyperparameter_search_results.csv', index=False)
    logger.info("Results saved to 'hyperparameter_search_results.csv'.")

    # Optionally, log the summary table to WandB
    wandb.log({"summary_table": wandb.Table(dataframe=results_df)})
    logger.info("Results logged to WandB.")

# Main function
def main():
    # Initialize WandB logging
    wandb.init(project="search_space_phinet", entity="mariam-jamal001-fhd")

    # Perform the sampling and compute the number of parameters and MACs
    num_samples = 1000
    params_vals, mac_vals, alpha_vals, beta_vals, t_zero_vals, num_layers_vals = sample_and_compute(num_samples)

    # Generate and visualize histogram of number of parameters
    plot_params_histogram(params_vals)

    # Visualize the results
    plot_results(alpha_vals, beta_vals, t_zero_vals, num_layers_vals, params_vals, mac_vals)

    # Save the results to CSV and log to WandB
    save_results_to_csv_and_wandb(alpha_vals, beta_vals, t_zero_vals, num_layers_vals, params_vals, mac_vals)

    # Finish the WandB run
    wandb.finish()
    logger.info("WandB run finished.")

# Entry point for the script
if __name__ == "__main__":
    try:
        logger.info("Script started.")
        main()
        logger.info("Script finished successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
