import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def correlation_analysis(proxy_scores, accuracy, loss):
    """
    Analyze correlations between proxy scores and model performance.
    Args:
        proxy_scores (list): List of proxy scores.
        accuracy (list): List of accuracies for the models.
        loss (list): List of validation losses for the models.
    Returns:
        dict: Correlation metrics and plots.
    """
    results = {}

    # Compute Pearson and Spearman correlations
    results['pearson_accuracy'] = pearsonr(proxy_scores, accuracy)
    results['spearman_accuracy'] = spearmanr(proxy_scores, accuracy)
    results['pearson_loss'] = pearsonr(proxy_scores, loss)
    results['spearman_loss'] = spearmanr(proxy_scores, loss)

    # Plot proxy scores vs accuracy/loss
    df = pd.DataFrame({'Proxy Score': proxy_scores, 'Accuracy': accuracy, 'Loss': loss})
    sns.pairplot(df, kind="scatter", diag_kind="kde")
    plt.show()

    return results
