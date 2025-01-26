# filename: plot_stock_price.py
import matplotlib.pyplot as plt
import yfinance as yf

# Define the tickers for NVDA and TESLA
tickers = ["NVDA", "TSLA"]

# Fetch the historical stock price data for the tickers
data = yf.download(tickers, start="2021-01-01")

# Calculate the YTD percentage change for each stock
data["YTD Percentage Change"] = (data["Close"] / data["Close"].iloc[0] - 1) * 100

# Plot the stock price change YTD
data["YTD Percentage Change"].plot(kind="bar")
plt.xlabel("Ticker")
plt.ylabel("YTD Percentage Change (%)")
plt.title("NVDA and TESLA Stock Price Change YTD")
plt.show()
