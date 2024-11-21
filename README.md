
# Stock Price Prediction with Moving Averages and LSTM

This project predicts stock prices using a Long Short-Term Memory (LSTM) model, leveraging historical data and technical indicators like moving averages. It uses deep learning techniques to forecast future prices and provides visualizations to evaluate the model's performance.

---

## Features

1. Fetches historical stock data using the [Yahoo Finance API](https://pypi.org/project/yfinance/).
2. Adds moving averages (short-term and long-term) as technical indicators.
3. Preprocesses data by normalizing and creating sequences for LSTM training.
4. Trains an LSTM model with early stopping to prevent overfitting.
5. Predicts stock prices for both test data and future periods (30 days).
6. Visualizes actual prices, predictions, and future forecasts.

---

## Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - Data Handling: `pandas`, `numpy`
  - Visualization: `matplotlib`
  - Stock Data: `yfinance`
  - Machine Learning: `scikit-learn`, `tensorflow`

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

1. Execute the script:
   ```bash
   python main.py
   ```

2. Input the stock ticker symbol (e.g., `AAPL` for Apple Inc.) when prompted.

---

## Code Workflow

1. **Data Fetching**: Downloads stock data for the given ticker and time range.
2. **Feature Engineering**: Computes short-term and long-term moving averages.
3. **Preprocessing**:
   - Normalizes features.
   - Prepares data sequences for LSTM training.
4. **Model Training**:
   - Constructs an LSTM model.
   - Trains the model using 80% of the data with early stopping.
5. **Prediction and Visualization**:
   - Predicts test set prices.
   - Forecasts future prices (30 days).
   - Generates plots for actual, predicted, and forecasted prices.

---

## Example Outputs

- **Historical Prices vs Predicted Prices**
  - A plot showing actual stock prices alongside LSTM predictions.
  
- **Future Predictions**
  - Dashed lines representing predicted prices for the next 30 business days.

---

## Customization

- Adjust the moving average windows by modifying `short_window` and `long_window` in the `add_moving_averages` function.
- Change the sequence length (`seq_length`) in the `preprocess_data` function for different look-back periods.
- Modify the training parameters (e.g., epochs, batch size) in the `train_and_evaluate` function.

---

## Limitations

- The model assumes that past stock trends will continue in the future.
- Performance depends on the chosen features and hyperparameters.
- External factors (e.g., market news) are not considered.

---

## Contributions

Feel free to fork this repository, make improvements, and submit pull requests.

---

