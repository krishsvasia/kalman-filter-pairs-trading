import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

class KalmanFilter:
    def __init__(self, delta = 1e-6, R = 0.6):

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

        return self.state

# Execution block
if __name__ == "__main__":
    # Make sure 'sp500_stocks.csv' is in your project folder!
    data = ingest_market_data('sp500_stocks.csv')
    
    # Instantiate the Kalman Filter.
    # FIXED: Added parentheses () to actually create the object instance.
    kf = KalmanFilter() 

    # We will keep track of the spread history to compute z-scores.
    spread_history = []
    current_position = 0
    # We will store the generated z-scores here because we may want to analyze them later.
    z_scores = []

    # Initialize a list to keep track of profit and loss (PnL) over time.
    pnl = [0]

    for i in range(len(data)):
        x_price = data["x_price"][i]
        y_price = data["y_price"][i]

        state = kf.update(x_price, y_price)
        slope, intercept = state[0], state[1]
    
        # We now calculate where we expect the y_price to be, given the x_price and the Kalman Filter's state.
        y_prediction = (slope * x_price) + intercept
        current_spread = y_price - y_prediction
        spread_history.append(current_spread)

        # We use 30 days of spread history to calculate the z-score.
        # The z-score tells us how many standard deviations away the current spread is from the mean, indicating if it's unusually high or low.
        window = 30 
        if len(spread_history) >= window:
            # Look at only the last 30 entries
            recent_window = spread_history[-window:]
            z_score = (current_spread - np.mean(recent_window)) / np.std(recent_window)
        else:
            # We don't have enough data to calculate a meaningful z-score we can act upon yet.
            z_score = 0
        z_scores.append(z_score)

        if i > 0:
            # Change in the spread value
            spread_change = spread_history[i] - spread_history[i-1]
            # Profit = direction of our bet * change in spread
            daily_profit = current_position * spread_change
            pnl.append(pnl[-1] + daily_profit)


        if z_score > 2:
            # This is when y is significantly overpriced compared to x.
            # We would short y and buy x.
            current_position = -1
        elif z_score < -2:
            # This is when y is significantly underpriced compared to x.
            # We would buy y and short x.
            current_position = 1
        elif abs(z_score) < 0.5:
            # This is when the spread has returned to normal.
            # Exit current position.
            current_position = 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Graph 1: The Z-Score
    ax1.plot(data['Date'], z_scores, color='blue', lw=1, label='Z-Score')
    ax1.axhline(2, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(-2, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(0, color='black', lw=1)
    ax1.set_title('Strategy Signals (Z-Score)')
    ax1.legend()

    # Graph 2: Cumulative PnL
    ax2.plot(data['Date'], pnl, color='green', lw=2, label='Cumulative Profit')
    ax2.set_title('Total Strategy Returns')
    ax2.set_ylabel('Profit ($)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"Final Profit: ${pnl[-1]:.2f}")

    total_return = pnl[-1] 
    # Assuming a starting capital of $100 for a percentage estimate.
    percent_return = (total_return / 100) * 100 

    # Calculate Sharpe Ratio.
    daily_returns = np.diff(pnl)
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

    # Count how many times you actually entered a trade.
    trade_count = np.sum(np.diff(np.array(z_scores) > 2).astype(int) + np.diff(np.array(z_scores) < -2).astype(int))

    print(f"Total Profit:        ${total_return:.2f}")
    print(f"Annualized Sharpe:   {sharpe:.2f}")
    print(f"Approx. Trade Count: {trade_count}")