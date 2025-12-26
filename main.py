import pandas as pd
import numpy as np

# Ingests raw market data and returns a 2D dataframe of close prices for two specified tickers.
def ingest_market_data(file_path, ticker_x = "ABT", ticker_y = "ABBV"):

    # Load S&P 500 stocks data.
    # A dataframe is a 2D array with improved functionality.
    df = pd.read_csv(file_path)

    # Converts the 'Date' column from Strings to DateTime objects.
    # This allows for the contents of the 'Date' column to be used in analysis.
    df['Date'] = pd.to_datetime(df['Date'])

    # Filters out data for only two columns: 'Date' and 'Close' for ticker_x and ticker_y.
    # This is done through the use of a mask.
    x_data = df[df['Symbol'] == ticker_x][['Date', 'Close']].rename(columns={'Close': 'x_price'})
    y_data = df[df['Symbol'] == ticker_y][['Date', 'Close']].rename(columns={'Close': 'y_price'})

    # We use an Inner Join to combine the two dataframes on the 'Date' column.
    # We drop rows where ticker_x has a price on days where ticker_y does not have a price, and vice versa.
    pair_df = pd.merge(x_data, y_data, on='Date', how='inner')

    # This removes the rows with any NaN values.
    pair_df.dropna(inplace=True)

    # Sort the rows by Date.
    # Inplace modifies the existing dataframe instead of creating a new one.
    pair_df.sort_values(by='Date', inplace=True)

    # Due to some rows being dropped, the index may be out of order.
    # Reset the index to be in order.
    return pair_df.reset_index(drop=True)

# Execution block
if __name__ == "__main__":
    # Make sure 'sp500_stocks.csv' is in your project folder!
    data = ingest_market_data('sp500_stocks.csv')
    
    print("Step 1 Complete!")
    print(f"Shape of the resulting 2D DataFrame: {data.shape}") # (Rows, Columns)
    # Display the first 5 rows with their named axes.
    print(data.head())

    class KalmanFilter:
        def __init__(self, delta = 1e - 5, R = 1e - 3):

            # A vector is created to hold the state.
            # self.state is a 2D vector where: [slope, intercept]
            self.state = np.zeros(2)
            # The initial slope is set to 1.0, assuming the stocks move 1-to-1.
            self.state[0] = 1.0

            # P is the state covariance matrix, and can be described as the "uncertainty" of the state.
            self.P = np.eye(2)
            # Q is the process noise covariance matrix, representing the uncertainty in the model.
            self.Q = delta / (1 - delta) * np.eye(2)
            # R is the measurement noise covariance, representing the uncertainty in the measurements.
            self.R = R

            # P and Q are 2x2 matrices, while R is a scalar.
            # P and Q represent two dimensions (slope and intercept), while R represents one dimension (the price).

            # This is the Kalman Gain, which will be computed during the update step.
            # It is not required during initialization, so we set it to None, and I am including it here for clarity.
            self.K = None

        def update(self, x_price, y_price):
            # Translates the state we created earlier into a measurement prediction.
            # H can be thought of as a prediction engine, mapping a prediction when input with x_price.
            H = np.array([[x_price, 1.0]])

            # We add the relationship noise to the state covariance.
            self.P += self.Q

            # Predicts a value for y_price based on the current state.
            y_predict = H @ self.state
            err = y_price - y_predict

            # Calculate the Kalman Gain.
            # This decides how much we trust the measurements.
            S = H @ self.P @ H.T + self.R
            self.K = self.P @ H.T / S

            # Update the slope and intercept based on the measurement error.
            self.state = self.state + (self.K.flatten() * err)

            # While Q increases the uncertainty, P, the Klarman Gain reduces it.
            # This is where the algorithm becomes more confident.
            self.P = self.P - (self.K @ H @ self.P)