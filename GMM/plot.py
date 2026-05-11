import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the folders and seed
seed = 3
folders = [f'gmm_CoCo_{seed}', f'gmm_IRM_{seed}', f'gmm_ERM_{seed}', f'gmm_REx_{seed}',f'gmm_DCIL_{seed}']
base_path = os.path.join('.', 'results')  # Path to 'results' folder

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 22,
})

# Initialize lists to store data
all_data = []

# Load data from each folder
for folder in folders:
    path = os.path.join(base_path, folder)
    with open(os.path.join(path, f'gmm_{seed}.pkl'), 'rb') as f:
        iter_r, train_r, test_r = pickle.load(f)

    # Get experiment name (CI, CoCo, etc.)
    experiment_name = folder.split('_')[1]


    # Create combined label: e.g., CoCo_Training, CoCo_Testing
    group_labels = [f'{experiment_name}_Training'] * len(iter_r) + [f'{experiment_name}_Testing'] * len(iter_r)

    data = {
        'Epoch': np.concatenate([iter_r, iter_r]),
        'Mean': np.concatenate([train_r[:, 0], test_r[:, 0]]),
        'Std': np.concatenate([train_r[:, 1], test_r[:, 1]]),
        'Group': group_labels,
        'Method': [experiment_name] * (2 * len(iter_r)),
        'Type': ['Training'] * len(iter_r) + ['Testing'] * len(iter_r)
    }

    df = pd.DataFrame(data)
    all_data.append(df)

# Combine into one DataFrame
combined_df = pd.concat(all_data, ignore_index=True)

# Set color palette for each method
palette = {
    'CoCo': '#ff7f0e',
    'IRM': '#2ca02c',
    'ERM': '#d62728',
    'REx': '#9467bd',
    'DCIL': '#1f77b4',#'DropCC': '#1f77b4',
}

# Plot
plt.figure(figsize=(14, 7))

# Draw lines with different dashes based on Type
for method in combined_df['Method'].unique():
    for ttype in ['Training', 'Testing']:
        subset = combined_df[(combined_df['Method'] == method) & (combined_df['Type'] == ttype)]
        linestyle = '-' if ttype == 'Training' else '--'
        label = f'{method}_{ttype}'
        plt.plot(subset['Epoch'], subset['Mean'],
                 label=label, linestyle=linestyle,
                 color=palette[method], linewidth=2)

        # Fill error band
        plt.fill_between(subset['Epoch'],
                         subset['Mean'] - subset['Std'],
                         subset['Mean'] + subset['Std'],
                         alpha=0.2, color=palette[method])

# Final touches
#plt.title('Training and Testing Accuracy of Each Method')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend(
    title=None,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=4,
    frameon=False
)
plt.tight_layout()

# Save figures
jpg_path = os.path.join(base_path, 'combined_performance_plot.jpg')
svg_path = os.path.join(base_path, 'combined_performance_plot.svg')

plt.savefig(jpg_path, dpi=300, bbox_inches='tight')
print("Saved JPG to:", jpg_path)

plt.savefig(svg_path, bbox_inches='tight')
print("Saved SVG to:", svg_path)

plt.close()
