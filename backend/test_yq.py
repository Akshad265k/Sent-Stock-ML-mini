from yahooquery import Ticker
import pandas as pd

def test_yq(ticker):
    print(f"Testing {ticker} with yahooquery...")
    t = Ticker(ticker)
    history = t.history(period="6mo")
    if isinstance(history, pd.DataFrame):
        print(f"Rows: {len(history)}")
        if len(history) > 0:
            print(history.tail())
    else:
        print(f"Error or No Data: {history}")

test_yq("TATAMOTORS.NS")
test_yq("TATAMOTORS.BO")
test_yq("AAPL")
