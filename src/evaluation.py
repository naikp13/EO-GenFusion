from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(ground_truth, predictions):
    """
    Evaluate model performance using MSE and R² score.
    Args:
        ground_truth (ndarray): True target values
        predictions (ndarray): Predicted values
    """
    mse = mean_squared_error(ground_truth, predictions)
    r2 = r2_score(ground_truth, predictions)
    print(f"Mean squared error: {mse:.3f}")
    print(f"R2 Variance score: {r2:.3f}")