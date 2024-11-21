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
