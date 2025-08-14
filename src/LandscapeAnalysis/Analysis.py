import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_feature_names_by_prefixes(prefixes, db_path = "data/landscape-analysis.db"):
    con = sqlite3.connect(db_path)
    cursor = con.execute("PRAGMA table_info('loss_surfaces')")
    cols = [
        row[1]
        for row in cursor
        if any(row[1].startswith(prefix) for prefix in prefixes)
        and not row[1].endswith("costs_runtime") # not run_time costs
    ]
    con.close()
    return cols

def get_non_null_columns(db_path='data/landscape-analysis.db'):
    con = sqlite3.connect(db_path)
    cursor = con.cursor()

    # Step 1: Get all column names
    cursor.execute(f"PRAGMA table_info('loss_surfaces')")
    columns = [row[1] for row in cursor.fetchall()]

    # Step 2: Test each column for NULLs
    non_null_cols = []
    for col in columns:
        cursor.execute(f"SELECT COUNT(*) FROM loss_surfaces WHERE {col} IS NULL")
        null_count = cursor.fetchone()[0]
        if null_count == 0:
            non_null_cols.append(col)
    con.close()
    return non_null_cols

def plot_pca_projection_matrix(pca, df_features, n_components=3, top_n=20):
    """
    Plots the PCA projection matrix as a heatmap for the top contributing features.

    Parameters:
    - pca: PCA object after fitting the data.
    - df_features: DataFrame containing the features used for PCA.
    - n_components: Number of principal components to include in the heatmap (default: 3).
    - top_n: Number of top contributing features to display (default: 20).
    """
    projection_matrix = pca.components_
    
    # Ensure n_components does not exceed the number of components in PCA
    n_components = min(n_components, projection_matrix.shape[0])
    
    # Get absolute values to measure importance
    abs_components = np.abs(projection_matrix[:n_components])
    
    # Find the most important features across all components
    feature_importance = np.sum(abs_components, axis=0)
    top_indices = np.argsort(feature_importance)[-top_n:]
    top_features = [df_features.columns[i] for i in top_indices]
    
    # Create matrix with only important features
    important_matrix = projection_matrix[:n_components, top_indices]
    
    # Create dataframe for heatmap
    heatmap_df = pd.DataFrame(
        important_matrix.T,
        index=top_features,
        columns=[f'Component {i+1}' for i in range(n_components)]
    )
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))  # Reduced figure size
    sns.heatmap(heatmap_df, cmap='coolwarm', center=0, annot=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('PCA Projection Matrix - Top Features', fontsize=10)
    plt.ylabel('Features', fontsize=9)
    plt.xlabel('Principal Components', fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()