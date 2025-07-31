# Generative Fusion of Synthetic Aperture Radar and Multispectral Data

This project implements a Deep Variational Autoencoder (VAE) with a probabilistic regressor for feature generation and prediction using TensorFlow and Keras. The model is trained with 5-fold cross-validation and evaluates performance using mean squared error and R² score. The code is modularized for better maintainability.

## Prerequisites

- Python 3.8+
- Install dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data as `.npy` files for features (`X`) and target (`Y`).
2. Update the `.npy` file paths in `src/data_processing.py` and `src/training.py` to point to your data.
3. Run the main script:
   ```bash
   python main.py
   ```

The script will:
- Load and standardize the data
- Build and train the VAE model with 5-fold cross-validation
- Save predictions to a `.npy` file
- Print the mean squared error and R² score

## Project Structure

- `main.py`: Main script to orchestrate the workflow
- `src/`
  - `__init__.py`: Package initialization
  - `data_processing.py`: Handles data loading and preprocessing
  - `model.py`: Defines and builds the VAE model
  - `training.py`: Manages training with cross-validation
  - `evaluation.py`: Evaluates model performance
- `requirements.txt`: List of Python dependencies
- `.gitignore`: Specifies files and directories to ignore in version control
- `README.md`: Project documentation

## Notes

- The model assumes input data is stored in `.npy` format.
- The script uses a fixed random seed for reproducibility.
- Training is performed silently (`verbose=0`) to reduce console output.

## License

This project is licensed under the MIT License.
