import numpy as np
from src.data_processing import load_and_preprocess_data
from src.model import build_vae_model
from src.training import train_with_cross_validation
from src.evaluation import evaluate_model

def main():
    # Set random seed for reproducibility
    np.random.seed(0)

    # Load and preprocess data
    X, Y, original_dim, num_train = load_and_preprocess_data()

    # Build VAE model
    vae, encoder, decoder = build_vae_model(original_dim)

    # Train with cross-validation
    pred = train_with_cross_validation(vae, encoder, X, Y, num_train)

    # Evaluate model
    evaluate_model(Y, pred)

if __name__ == "__main__":
    main()