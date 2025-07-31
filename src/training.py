import numpy as np
from sklearn.model_selection import StratifiedKFold

def train_with_cross_validation(vae, encoder, X, Y, num_train, batch_size=10, epochs=1000):
    """
    Train the VAE model with 5-fold cross-validation.
    Args:
        vae (Model): VAE model
        encoder (Model): Encoder model
        X (ndarray): Features
        Y (ndarray): Target
        num_train (int): Number of training samples
        batch_size (int): Batch size for training
        epochs (int): Number of epochs
    Returns:
        pred (ndarray): Predictions
    """
    skf = StratifiedKFold(n_splits=5)
    pred = np.zeros((Y.shape))
    fake = np.zeros((Y.shape[0]))
    fake[:num_train] = 1

    for train_idx, test_idx in skf.split(X, fake):
        training_feature_sk = X[train_idx, :]
        training_score = Y[train_idx]
        testing_feature_sk = X[test_idx, :]
        testing_score = Y[test_idx]

        vae.load_weights('weights.h5')
        vae.fit([training_feature_sk, training_score],
                epochs=epochs,
                batch_size=batch_size,
                verbose=0)

        [z_mean, z_log_var, z, r_mean, r_log_var, r_vae, pz_mean] = encoder.predict(
            [testing_feature_sk, testing_score], batch_size=batch_size)
        pred[test_idx] = r_mean[:, 0].reshape(-1, 1)
        np.save('.npy', pred)  # Update path as needed

    return pred