import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']  # Example stock tickers


# Loop through each ticker and calculate momentum
momentum_dict = {}
period = 126
for ticker in tickers:
    raw_data = yf.download(ticker, start="2015-01-01", end="2024-12-31")['Adj Close']
    data = raw_data.values
    momentum = data[- 1] / data[-period] - 1
    momentum_dict[ticker] = momentum

# Convert dictionary to DataFrame for easier visualisation
momentum_df = pd.DataFrame(list(momentum_dict.items()), columns=['Ticker', 'Momentum'])
momentum_df['Momentum (%)'] = momentum_df['Momentum'] * 100
momentum_df['Momentum (%)'] = momentum_df['Momentum'].astype(float)

print(momentum_df['Momentum (%)'])


# Plot the momentum for each stock
momentum_df.set_index('Ticker', inplace=True)
momentum_df['Momentum (%)'].plot(kind='bar', color='skyblue', title="Stock Momentum")
plt.ylabel('Momentum (%)')
plt.show()
