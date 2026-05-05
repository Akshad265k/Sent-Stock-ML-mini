import yfinance as yf
ticker = "TATAMOTORS.NS"
s = yf.Ticker(ticker)
h = s.history(period="1mo")
print(f"1mo: {len(h)}")
h2 = s.history(period="1y")
print(f"1y: {len(h2)}")
h3 = s.history(start="2024-01-01", end="2024-05-01")
print(f"Specific range: {len(h3)}")
