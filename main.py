import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential   # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout   # type: ignore
from tensorflow.keras.callbacks import EarlyStopping   # type: ignore

# Step 1: Fetch the stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Step 2: Add moving averages
def add_moving_averages(data, short_window=10, long_window=50):
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    # Drop rows with NaN values due to moving averages
    data.dropna(inplace=True)
    return data

# Step 3: Preprocess the data
def preprocess_data(data, features=['Close', 'Short_MA', 'Long_MA']):
    # Select the relevant features
    dataset = data[features].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create sequences for LSTM
    X, y = [], []
    seq_length = 60  # Using 60 days of data to predict the next day's price
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, :])  # Use all features in the sequence
        y.append(scaled_data[i, 0])  # Predict the 'Close' price
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

# Step 4: Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),  # Adding dropout to prevent overfitting
        LSTM(50, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 5: Train and evaluate
import matplotlib.dates as mdates

def train_and_evaluate(X_train, y_train, X_test, y_test, scaler, actual_data, stock_data):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train, 
        validation_split=0.2, 
        batch_size=32, 
        epochs=50, 
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], X_test.shape[2] - 1)))))[:, 0]  # Reverse scaling for 'Close'
    
    # Reverse scale the test set
    actual_test = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X_test.shape[2] - 1)))))[:, 0]
    
    # Predict the next 30 days
    future_predictions = []
    last_sequence = X_test[-1]  # Start with the last available sequence
    for _ in range(30):
        next_pred = model.predict(last_sequence[np.newaxis, :, :])[0, 0]
        future_predictions.append(next_pred)
        
        # Update the sequence with the new prediction
        new_sequence = np.roll(last_sequence, -1, axis=0)
        new_sequence[-1, 0] = next_pred  # Update only the 'Close' feature
        last_sequence = new_sequence

    # Scale future predictions back to the original range
    future_predictions = scaler.inverse_transform(np.hstack((
        np.array(future_predictions).reshape(-1, 1),
        np.zeros((30, X_test.shape[2] - 1))
    )))[:, 0]
    
    # Plot results
    plt.figure(figsize=(14, 6))

    # Plot actual prices (train and test combined)
    plt.plot(stock_data.index, actual_data, label='Actual Price (Train/Test)', alpha=0.6)

    # Calculate indices for predictions
    prediction_start = len(actual_data) - len(y_test)
    prediction_end = prediction_start + len(predictions)

    # Plot predictions aligned to the actual test prices
    plt.plot(stock_data.index[prediction_start:prediction_end], predictions, label='Predicted Price', color='red')

    # Calculate indices for future predictions
    future_dates = pd.date_range(stock_data.index[-1], periods=30, freq='B')
    future_start = prediction_end
    future_end = future_start + len(future_predictions)

    # Plot future predictions
    plt.plot(future_dates, future_predictions, label='Future Predictions', color='green', linestyle='dashed')

    # Format x-axis with larger year labels
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())  # Major ticks at the start of each year
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))  # Minor ticks at April, July, October
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Major ticks show years
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))  # Minor ticks show months

    # Customize tick label fonts
    for label in ax.get_xticklabels(which='major'):
        label.set_fontsize(11)  # Larger font size for year labels
        label.set_fontweight('bold')  # Bold year labels
    for label in ax.get_xticklabels(which='minor'):
        label.set_fontsize(8)  # Smaller font size for month labels

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add title, labels, and legend
    plt.title('Stock Price Prediction with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# Main function
if __name__ == "__main__":
    # Input parameters
    ticker = input('Enter the symbol of the stock ') 
    start_date = "2015-01-01"
    end_date = "2024-11-21"
    
    # Load and preprocess data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    stock_data = add_moving_averages(stock_data)  # Add moving averages
    X, y, scaler = preprocess_data(stock_data)
    
    # Split into training and testing
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    actual_data = stock_data['Close'].values
    
    # Train and evaluate the model
    train_and_evaluate(X_train, y_train, X_test, y_test, scaler, actual_data, stock_data)
