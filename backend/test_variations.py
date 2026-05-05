import yfinance as yf

def test_search(query):
    print(f"Searching for: {query}")
    try:
        # Some versions of yfinance have a Search class or similar
        # But usually we use the Ticker object
        # Let's try to find if there is a way to get ticker from name
        pass
    except Exception as e:
        print(f"Error: {e}")

# Test with various common Indian stocks
tickers = ["TATAMOTORS.NS", "TATA-MOTORS.NS", "TATAMTR.NS"]
for t in tickers:
    s = yf.Ticker(t)
    h = s.history(period="1d")
    print(f"{t}: {len(h)} rows")
