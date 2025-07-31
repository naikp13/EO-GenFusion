import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    """
    Load and preprocess data from .npy files, applying standardization.
    Returns:
        X (ndarray): Standardized features
        Y (ndarray): Standardized target
        original_dim (int): Feature dimension
        num_train (int): Number of training samples
    """
    # Load data
    training_feature = np.load('.npy')  # Update path as needed
    X = training_feature
    Y = np.load('.npy')  # Update path as needed
    ground_truth_r = Y

    # Scaling of data
    PredictorScaler = StandardScaler()
    TargetVarScaler = StandardScaler()

    PredictorScalerFit = PredictorScaler.fit(X)
    TargetVarScalerFit = TargetVarScaler.fit(Y.reshape(-1, 1))

    # Generating standardized values
    X = PredictorScalerFit.transform(X)
    Y = TargetVarScalerFit.transform(Y.reshape(-1, 1))

    original_dim = X.shape[1]
    num_train = X.shape[0]

    return X, Y, original_dim, num_train