# Bitcoin Price Prediction Project

This project provides a standard Python package for predicting Bitcoin prices using machine learning models, including LSTM and Transformer-based neural networks. It is based on the workflow and code from the `BitcoinPricePrediction_LSTM.ipynb` notebook.

## Features
- Data preprocessing and feature engineering for Bitcoin price data
- Multiple regression models (Linear Regression, Random Forest, Gradient Boosting, Decision Tree)
- LSTM-based deep learning model for time series prediction
- Transformer-based deep learning model for advanced sequence modeling
- Visualization of results and model performance

## Project Structure
- `data/` - Place your raw data files here (e.g., `btcusd_1-min_data.csv`)
- `notebooks/` - Jupyter notebooks for exploration and prototyping
- `src/` - Main source code for data processing, modeling, and prediction
- `tests/` - Unit tests for the project
- `models/` - Saved model weights and scalers

## Getting Started
1. Clone the repository or copy the project files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the Bitcoin price data CSV in the `data/` directory.
4. Run the main scripts for training and prediction.

## Requirements
- Python 3.8+
- See `requirements.txt` for full list

## Usage

### LSTM Model
- **Train the LSTM model:**
  ```bash
  python src/train_lstm.py
  ```
- **Predict the next day's price with LSTM:**
  ```bash
  python src/predict_lstm.py
  ```

### Transformer Model
- **Train the Transformer model:**
  ```bash
  python src/train_transformer.py
  ```
- **Predict the next day's price with Transformer:**
  ```bash
  python src/predict_transformer.py
  ```

## Notes
- The data file `btcusd_1-min_data.csv` should be placed in the `data/` directory.
- Trained models and scalers are saved in the `models/` directory.
- The project is based on the logic and code from the included Jupyter notebook for reproducibility and further exploration.

## License
MIT License 