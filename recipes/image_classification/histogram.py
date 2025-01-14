import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/home/majam001/thesis_mariam/micromind/recipes/image_classification/hyperparameter_search_results.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Check if the 'params' column exists
if 'params' in data.columns:
    # Filter the 'params' column for values between 0 and 100
    filtered_params = data['params'][(data['params'] >= 0) & (data['params'] <= 0.5)]
    
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(filtered_params, bins=10, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Params', fontsize=16)
    plt.xlabel('Params (M)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('histogram_0.5.png')
else:
    print("The 'params' column is not found in the CSV file.")
