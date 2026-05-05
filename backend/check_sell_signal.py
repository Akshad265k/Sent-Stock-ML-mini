import yfinance as yf

def check_signal(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")
    if len(hist) < 50:
        return "Not enough data"
    
    price = hist['Close'].iloc[-1]
    ma50 = hist['Close'].rolling(50).mean().iloc[-1]
    
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
    
    print(f"Ticker: {ticker}")
    print(f"Price: {price:.2f}, MA50: {ma50:.2f}, RSI: {rsi:.2f}")
    if price < ma50 and rsi > 30:
        return "SELL (Technical)"
    else:
        return "Not a clear SELL"

print(check_signal("LUV"))
print(check_signal("CCI"))
