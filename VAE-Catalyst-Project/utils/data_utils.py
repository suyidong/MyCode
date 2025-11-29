import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, TensorDataset
import torch

def fps_sample(X, ratio=0.8, metric='euclidean', random_state=None):
    """
    Farthest Point Sampling for dataset splitting.
    
    Args:
        X (ndarray): Input data
        ratio (float): Sampling ratio
        metric (str): Distance metric
        random_state (int): Random seed
        
    Returns:
        ndarray: Indices of selected samples
    """
    np.random.seed(random_state)
    n_total = len(X)
    n_sample = int(n_total * ratio)

    if n_sample >= n_total:
        return np.arange(n_total)

    # Randomly initialize first point
    idx_all = np.arange(n_total)
    select_idx = [np.random.choice(idx_all)]

    # Initialize distance vector: minimum distance to selected set
    dist = pairwise_distances(X, X[select_idx[-1]:select_idx[-1] + 1], metric=metric).ravel()
    dist[select_idx[-1]] = -1  # Mark selected points as -1

    while len(select_idx) < n_sample:
        # Select point with maximum distance (excluding selected points)
        farthest = np.argmax(dist)
        select_idx.append(farthest)
        dist[farthest] = -1  # Mark as selected

        # Update minimum distances
        if len(select_idx) < n_sample:
            new_dist = pairwise_distances(X, X[farthest:farthest + 1], metric=metric).ravel()
            dist = np.where(dist == -1, -1, np.minimum(dist, new_dist))

    return np.array(select_idx)

def prepare_data(data_path, target_column='Target', fps_ratio=0.48, random_state=1895):
    """
    Prepare training and test data using FPS sampling.
    
    Args:
        data_path (str): Path to data file
        target_column (str): Name of target column
        fps_ratio (float): Ratio for FPS sampling
        random_state (int): Random seed
        
    Returns:
        tuple: (train_loader, X_train_tensor, y_train_tensor, X_test_tensor, 
                y_test_tensor, x_scaler, y_scaler, input_dim)
    """
    # Load data
    data = pd.read_excel(data_path)
    
    # Separate features and target
    X = data.drop(target_column, axis=1).values
    y = data[target_column].values.reshape(-1, 1)
    X_vae = X[:, :2]  # Use first two features for VAE

    # FPS sampling
    fps_normalized = StandardScaler()
    X_normalized = fps_normalized.fit_transform(X)
    fps_idx = fps_sample(X_normalized, ratio=fps_ratio, metric='euclidean', random_state=random_state)
    
    train_mask = np.zeros(len(X), dtype=bool)
    train_mask[fps_idx] = True
    test_mask = ~train_mask

    # Standardize data using only training set
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_raw = X_vae[train_mask]
    y_train_raw = y[train_mask]
    X_test_raw = X_vae[test_mask]
    y_test_raw = y[test_mask]

    x_scaler.fit(X_train_raw)
    y_scaler.fit(y_train_raw)

    X_train = x_scaler.transform(X_train_raw)
    y_train = y_scaler.transform(y_train_raw)
    X_test = x_scaler.transform(X_test_raw)
    y_test = y_scaler.transform(y_test_raw)

    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    return (train_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            x_scaler, y_scaler, input_dim=X_vae.shape[1])
