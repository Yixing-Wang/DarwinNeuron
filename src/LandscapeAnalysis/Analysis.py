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

def plot_pca_projection_matrix(pca, df_features, n_components=3, top_n=20, plot_abs=False):
    """
    Plots the PCA projection matrix as a heatmap for the top contributing features.

    Parameters:
    - pca: PCA object after fitting the data.
    - df_features: DataFrame containing the features used for PCA.
    - n_components: Number of principal components to include in the heatmap (default: 3).
    - top_n: Number of top contributing features to display (default: 20).
    - plot_abs: If True, plot the absolute values of the projection matrix (default: False).
    """
    projection_matrix = pca.components_
    
    # Ensure n_components does not exceed the number of components in PCA
    n_components = min(n_components, projection_matrix.shape[0])
    
    # Get absolute values to measure importance (for selecting top features)
    abs_components = np.abs(projection_matrix[:n_components])
    
    # Find the most important features across all components
    feature_importance = np.sum(abs_components, axis=0)
    top_indices = np.argsort(feature_importance)[-top_n:]
    top_features = [df_features.columns[i] for i in top_indices]
    
    # Create matrix with only important features
    important_matrix = projection_matrix[:n_components, top_indices]
    
    # Optionally take absolute values for plotting
    plot_matrix = np.abs(important_matrix) if plot_abs else important_matrix
    
    # Create dataframe for heatmap (features as rows)
    heatmap_df = pd.DataFrame(
        plot_matrix.T,
        index=top_features,
        columns=[f'Component {i+1}' for i in range(n_components)]
    )
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_df, cmap='coolwarm', center=0 if not plot_abs else None, annot=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    title_suffix = ' (absolute values)' if plot_abs else ''
    plt.title(f'PCA Projection Matrix - Top Features{title_suffix}', fontsize=10)
    plt.ylabel('Features', fontsize=9)
    plt.xlabel('Principal Components', fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()
    return heatmap_df

#####################################################################
import sqlite3, pandas as pd, numpy as np, pickle

def update_blob():

    DB = "data/landscape-analysis.db"
    TABLE = "loss_surfaces"
    COL = "ela_distr_number_of_peaks"

    def decode_to_int(v):
        if v is None:
            return None
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return int(v)
        if isinstance(v, str):
            try:
                return int(float(v.strip()))
            except Exception:
                return None
        if isinstance(v, (bytes, bytearray, memoryview)):
            b = bytes(v)
            # try pickle first
            try:
                obj = pickle.loads(b)
                if isinstance(obj, (np.generic,)):
                    return int(np.asarray(obj).item())
                if isinstance(obj, (np.ndarray, list, tuple)):
                    return int(np.asarray(obj).ravel()[0])
                if isinstance(obj, (int, float)):
                    return int(obj)
            except Exception:
                pass
            # try raw 64-bit buffer (int, then float)
            for dt in ('<i8','<f8'):
                try:
                    arr = np.frombuffer(b, dtype=dt)
                    if arr.size == 1:
                        return int(arr[0])
                except Exception:
                    pass
            # last resort, try utf-8 digits
            try:
                return int(b.decode('utf-8').strip())
            except Exception:
                return None
        return None

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Pull rowid + column
    df = pd.read_sql_query(f"SELECT rowid AS _id, {COL} FROM {TABLE};", conn)

    # Decode and write back to the SAME column
    updates = []
    for _id, raw in zip(df["_id"], df[COL]):
        val = decode_to_int(raw)
        updates.append((val, int(_id)))

    cur.executemany(
        f"UPDATE {TABLE} SET {COL} = ? WHERE rowid = ?;",
        updates
    )
    conn.commit()
    conn.close()

    print("Done. Values written back to the original column.")