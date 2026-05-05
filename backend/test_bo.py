import yfinance as yf
ticker = "TATAMOTORS.BO"
stock = yf.Ticker(ticker)
hist = stock.history(period="1d")
print(f"{ticker}: {len(hist)} rows")
