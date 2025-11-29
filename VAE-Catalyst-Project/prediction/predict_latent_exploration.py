import torch
import numpy as np
import pandas as pd
import itertools
from models.vae_model import ImprovedVAE
from inference.predict_one import load_model

def traverse_latent_space_and_predict(model, x_scaler, y_scaler,
                                      latent_dim=5,
                                      points_per_dim=10,
                                      value_range=(-3, 3),
                                      target_range=(0, 0.3)):
    """
    Traverse latent space and make predictions, filtering results within target range.
    
    Args:
        model: Trained VAE model
        x_scaler: Feature scaler
        y_scaler: Target scaler
        latent_dim (int): Latent space dimension
        points_per_dim (int): Number of points per dimension
        value_range (tuple): Range for latent variable values
        target_range (tuple): Target prediction range for filtering
        
    Returns:
        list: List of valid results
    """
    device = next(model.parameters()).device
    
    # Generate latent space grid points
    linspace = np.linspace(value_range[0], value_range[1], points_per_dim)
    grid_points = list(itertools.product(linspace, repeat=latent_dim))

    print(f"Generated {len(grid_points)} latent space points...")

    results = []
    valid_count = 0

    for i, point in enumerate(grid_points):
        if i % 1000 == 0:
            print(f"Processing point {i}/{len(grid_points)}...")

        # Convert latent point to tensor
        z = torch.FloatTensor([point]).to(device)

        # Decode to get prediction and reconstruction
        with torch.no_grad():
            pred, reconstructed = model.decode(z)

        # Inverse transform
        pred_np = pred.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()

        pred_orig = y_scaler.inverse_transform(pred_np)[0][0]
        reconstructed_orig = x_scaler.inverse_transform(reconstructed_np)[0]

        # Filter results within target range
        if target_range[0] <= pred_orig <= target_range[1]:
            result = {
                'latent_vector': point,
                'prediction': pred_orig,
                'reconstructed_features': reconstructed_orig
            }
            results.append(result)
            valid_count += 1

    print(f"Found {valid_count} samples with predictions in range {target_range}")
    return results

def save_results_to_excel(results, filename='latent_space_results.xlsx'):
    """
    Save results to Excel file.
    
    Args:
        results (list): List of result dictionaries
        filename (str): Output filename
        
    Returns:
        DataFrame: Combined results DataFrame
    """
    # Prepare data
    latent_vectors = [result['latent_vector'] for result in results]
    predictions = [result['prediction'] for result in results]
    reconstructed_features = [result['reconstructed_features'] for result in results]

    # Create DataFrames
    latent_df = pd.DataFrame(latent_vectors, columns=[f'latent_dim_{i + 1}' for i in range(len(latent_vectors[0]))])
    pred_df = pd.DataFrame(predictions, columns=['prediction'])

    # Reconstruction features DataFrame
    n_features = len(reconstructed_features[0])
    recon_df = pd.DataFrame(reconstructed_features, columns=[f'feature_{i + 1}' for i in range(n_features)])

    # Combine all data
    final_df = pd.concat([latent_df, pred_df, recon_df], axis=1)

    # Save to Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        final_df.to_excel(writer, sheet_name='Filtered Results', index=False)

        # Add summary information
        summary_data = {
            'Total Samples': [len(results)],
            'Prediction Range': [f"{final_df['prediction'].min():.4f} - {final_df['prediction'].max():.4f}"],
            'Mean Prediction': [final_df['prediction'].mean()],
            'Standard Deviation': [final_df['prediction'].std()]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"Results saved to {filename}")
    return final_df

def main():
    """Main latent space exploration function."""
    # Load model
    model_path = 'vae_catalyst_model.pth'
    model, x_scaler, y_scaler = load_model(model_path)
    print("Model loaded successfully!")

    # Traverse latent space and filter results
    print("Starting latent space exploration...")
    results = traverse_latent_space_and_predict(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        latent_dim=5,
        points_per_dim=8,
        value_range=(-1.5, 1.5),
        target_range=(-0.3, 0.3)
    )

    if results:
        # Save results to Excel
        df = save_results_to_excel(results)

        # Print statistics
        print(f"\nResults Statistics:")
        print(f"Found {len(results)} valid samples")
        print(f"Prediction range: {df['prediction'].min():.4f} - {df['prediction'].max():.4f}")
        print(f"Mean prediction: {df['prediction'].mean():.4f}")

        # Show first few results
        print(f"\nFirst 5 results:")
        print(df.head())
    else:
        print("No valid samples found. Please adjust search range or sampling density.")

if __name__ == "__main__":
    main()
