import yfinance as yf
ticker = "TCS.NS"
s = yf.Ticker(ticker)
h = s.history(period="1mo")
print(f"TCS.NS: {len(h)}")
