import yfinance as yf
ticker = "AAPL"
stock = yf.Ticker(ticker)
hist = stock.history(period="6mo")
print(f"Ticker: {ticker}")
print(f"Data length: {len(hist)}")
if len(hist) > 0:
    print(hist.tail())
else:
    print("No data found")
