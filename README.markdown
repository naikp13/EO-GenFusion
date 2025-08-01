# Generative Fusion of Synthetic Aperture Radar and Multispectral Data
This is an implementation of the work pulished in the IEEE JSTARS paper that can be accessed here - https://doi.org/10.1109/jstars.2022.3179027


<img src="https://raw.githubusercontent.com/naikp13/GIMMEO/main/images/architecture.gif" alt="Model Architecture" width="400"/>

This project implements a Deep Variational Autoencoder (VAE) with a probabilistic regressor for feature generation and prediction using TensorFlow and Keras. The proposed generative consists of two main component networks—an encoder and a decoder. The additional two networks added to this traditional VAE architecture are — a latent generator and a regressor network. Unlike traditional approaches that separately trains a feed-forward regressor network, the proposed approach has an integrated regressor network connected to the reparametrized latent space via a latent generator network. All these networks are interassociated with a combination of three loss functions (Reconstruction loss, KL Loss and Label loss).
## Prerequisites

- Python 3.8+
- Install dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/naikp13/GIMMEO.git
   cd GIMMEO
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
- Save predictions
- Print the evaluation metrics (MAE and R² score)

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

## Cite this work

Please cite this article in case this method was helpful for your research or used for your work,

```Citation
Naik, P., Dalponte, M., & Bruzzone, L. (2022). Generative Feature Extraction From Sentinel 1 and 2 Data for Prediction of Forest Aboveground Biomass in the Italian Alps. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 15, 4755–4771. https://doi.org/10.1109/jstars.2022.3179027
```

## Contact

For issues or questions, open an issue on GitHub or contact [parthnaik1993@gmail.com](mailto:parthnaik1993@gmail.com).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
