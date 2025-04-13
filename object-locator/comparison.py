import pandas as pd
import matplotlib as plt
import numpy as np
from .metrics import Judge
import torch

def plot_bmm_otsu_comparison(csv_path, radii, title='BMM vs Otsu Comparison'):
    """
    Plot the performance comparison between BMM and Otsu methods as radius changes.
    Parameters:
        csv_path: Path to the CSV file containing evaluation metrics.
        radii: List of radii to analyze.
        title: Title of the chart.
    Returns:
        matplotlib Figure object.
    """
    # Read data
    df = pd.read_csv(csv_path)
    
    # Create the figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle(title, y=0.92, fontsize=14)
    
    # Prepare colors and marker styles
    colors = plt.cm.viridis(np.linspace(0, 1, len(radii)))
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']
    
    # Collect data
    bmm_precision = []
    bmm_recall = []
    bmm_fscore = []
    otsu_precision = []
    otsu_recall = []
    otsu_fscore = []
    actual_radii = []
    
    for r in radii:
        # Find the closest radius
        r_selected = df.r.values[np.argmin(np.abs(df.r.values - r))]
        actual_radii.append(r_selected)
        
        # Get BMM data (tau = -2)
        bmm_data = df[(df.r == r_selected) & (df.th == -2)]
        if not bmm_data.empty:
            bmm_precision.append(bmm_data.precision.values[0])
            bmm_recall.append(bmm_data.recall.values[0])
            bmm_fscore.append(bmm_data.fscore.values[0])
        
        # Get Otsu data (assuming tau = -1 represents Otsu)
        otsu_data = df[(df.r == r_selected) & (df.th == -1)]
        if not otsu_data.empty:
            otsu_precision.append(otsu_data.precision.values[0])
            otsu_recall.append(otsu_data.recall.values[0])
            otsu_fscore.append(otsu_data.fscore.values[0])
    
    # Plot Precision comparison
    ax1.plot(actual_radii[:len(bmm_precision)], bmm_precision, 
             'b-', marker='o', label='BMM Precision')
    ax1.plot(actual_radii[:len(otsu_precision)], otsu_precision, 
             'r--', marker='s', label='Otsu Precision')
    ax1.set_xlabel('Radius (pixels)')
    ax1.set_ylabel('Precision (%)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Recall comparison
    ax2.plot(actual_radii[:len(bmm_recall)], bmm_recall, 
             'b-', marker='o', label='BMM Recall')
    ax2.plot(actual_radii[:len(otsu_recall)], otsu_recall, 
             'r--', marker='s', label='Otsu Recall')
    ax2.set_xlabel('Radius (pixels)')
    ax2.set_ylabel('Recall (%)')
    ax2.grid(True)
    ax2.legend()
    
    # Plot F-score comparison
    ax3.plot(actual_radii[:len(bmm_fscore)], bmm_fscore, 
             'b-', marker='o', label='BMM F-score')
    ax3.plot(actual_radii[:len(otsu_fscore)], otsu_fscore, 
             'r--', marker='s', label='Otsu F-score')
    ax3.set_xlabel('Radius (pixels)')
    ax3.set_ylabel('F-score')
    ax3.grid(True)
    ax3.legend()
    
    return fig


def analyze_alpha_performance(model, val_loader, alphas, device):
    """
    Analyze the impact of different alpha values on model performance.
    Parameters:
        model: Trained model
        val_loader: Validation dataset loader
        alphas: List of alpha values to test
        device: Computing device
    Returns:
        DataFrame containing performance metrics for each alpha value
    """
    results = []
    original_p = model.module.loss_loc.p.item()  # Save the original alpha value
    
    for alpha in alphas:
        # Set the new alpha value
        model.module.loss_loc.p.data.fill_(alpha)
        
        model.eval()
        judge = Judge(r=5)  # Set the r value according to your requirements
        total_loss = 0
        
        with torch.no_grad():
            for imgs, dictionaries in val_loader:
                imgs = imgs.to(device)
                # Prepare target data...
                
                # Forward pass
                est_maps, est_counts = model(imgs)
                # Calculate metrics...
                
                # Record results
                judge.feed_points(...)
                judge.feed_count(...)
        
        # Collect metrics
        results.append({
            'alpha': alpha,
            'precision': judge.precision,
            'recall': judge.recall,
            'fscore': judge.fscore,
            'mahd': judge.mahd
        })
    
    # Restore the original alpha value
    model.module.loss_loc.p.data.fill_(original_p)
    
    return pd.DataFrame(results)