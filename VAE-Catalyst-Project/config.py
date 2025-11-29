# Training configuration
TRAIN_CONFIG = {
    'random_seed': 1895,
    'batch_size': 32,
    'learning_rate': 0.004,
    'epochs': 1000,
    'latent_dim': 5,
    'hidden_dims': [64, 32],
    'beta': 0.01,
    'recon_weight': 0.8,
    'scheduler_step': 300,
    'scheduler_gamma': 0.7
}

# Data configuration
DATA_CONFIG = {
    'fps_ratio': 0.48,
    'target_column': 'Target',
    'default_data_path': 'way/to/your/data/fulldata.xlsx'  # 添加默认数据路径
}

# Inference configuration
INFERENCE_CONFIG = {
    'latent_points_per_dim': 8,
    'latent_value_range': (-1.5, 1.5),
    'target_range': (-0.3, 0.3)
}
