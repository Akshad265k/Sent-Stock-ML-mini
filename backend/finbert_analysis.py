import yfinance as yf
from gnews import GNews
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd
import numpy as np
import csv
import os
from datetime import datetime

# ---------------------------------------------------------
# 1. GLOBAL SETUP (Load AI Model Once)
# ---------------------------------------------------------
print("\n🧠 INITIALIZING SYSTEM...")
print("   Loading FinBERT Financial Sentiment Model... (This may take a moment)")

# Load FinBERT (Pre-trained on financial text)
try:
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    # device=-1 uses CPU (change to 0 if you have a GPU set up with CUDA)
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer, device=-1)
    print("   ✅ Model Loaded Successfully.")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    exit()

# Score Mapping: Positive=1, Neutral=0, Negative=-1
score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def get_sentiment_score(headline):
    """
    Analyzes a single headline and returns the numerical score and label.
    """
    # Truncate to 512 tokens to match BERT limits
    results = nlp(headline[:512]) 
    result = results[0]
    label = result['label']
    score = score_map[label]
    return score, label

# def get_technical_signal(ticker):
#     """
#     Fetches price data and calculates Moving Average (50-day) and RSI (14-day).
#     """
#     try:
#         stock = yf.Ticker(ticker)
#         # Fetch 6 months of data to ensure enough points for moving averages
#         hist = stock.history(period="6mo")
        
#         if len(hist) < 50:
#             return "UNKNOWN (Insufficient Data)", 0, 0, 0

#         current_price = hist['Close'].iloc[-1]
        
#         # 1. Calculate 50-Day Simple Moving Average (SMA)
#         ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        
#         # 2. Calculate RSI (14-day)
#         delta = hist['Close'].diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#         rs = gain / loss
#         rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
#         # 3. Determine Technical Trend
#         signal = "NEUTRAL (Sideways)"
        
#         if current_price > ma_50:
#             if rsi < 70:
#                 signal = "BULLISH (Uptrend)"
#             else:
#                 signal = "NEUTRAL (Overbought - Risky)"
#         elif current_price < ma_50:
#             if rsi > 30:
#                 signal = "BEARISH (Downtrend)"
#             else:
#                 signal = "NEUTRAL (Oversold - Watch for Reversal)"
                
#         return signal, current_price, ma_50, rsi

#     except Exception as e:
#         print(f"   ❌ Technical Analysis Error: {e}")
#         return "ERROR", 0, 0, 0

def get_technical_signal(ticker):
    """
    Robust technical analysis that ALWAYS produces a price trend:
    - Uses MA50, fallback to MA20, fallback to EMA10
    - Uses RSI14, fallback to momentum
    - Works even with limited Yahoo data
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")

        if hist is None or len(hist) == 0:
            return "UNKNOWN (No Market Data Found)", 0, 0, 0

        # Clean data
        hist = hist[['Close']].dropna()
        hist['Close'] = hist['Close'].ffill().bfill()

        price = hist['Close'].iloc[-1]

        # -----------------------------
        # 1. Try Standard MA50
        # -----------------------------
        ma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None
        ma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else None
        ema_10 = hist['Close'].ewm(span=10).mean().iloc[-1]

        # Best available baseline
        if ma_50:
            baseline = ma_50
            baseline_type = "MA50"
        elif ma_20:
            baseline = ma_20
            baseline_type = "MA20"
        else:
            baseline = ema_10
            baseline_type = "EMA10"

        # -----------------------------
        # 2. Try Standard RSI14
        # -----------------------------
        if len(hist) >= 15:
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
        else:
            # Fallback RSI (momentum-based)
            rsi = ((price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100
            baseline_type += " + MomentumRSI"

        # -----------------------------
        # 3. Determine Trend
        # -----------------------------
        trend = "NEUTRAL"

        # Price vs baseline trend
        if price > baseline:
            if rsi < 70:
                trend = f"BULLISH (Above {baseline_type})"
            else:
                trend = f"NEUTRAL (Overbought at {baseline_type})"
        elif price < baseline:
            if rsi > 30:
                trend = f"BEARISH (Below {baseline_type})"
            else:
                trend = f"NEUTRAL (Oversold near {baseline_type})"

        return trend, price, baseline, rsi

    except Exception as e:
        return f"ERROR({e})",0,0,0

def log_prediction(ticker, sentiment, signal, verdict, price):
    """
    Logs the prediction to a CSV file for accuracy tracking.
    """
    filename = 'trading_journal.csv'
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(['Date', 'Ticker', 'Price', 'Sentiment_Status', 'Tech_Signal', 'Final_Verdict', 'Result'])
            
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            ticker,
            f"{price:.2f}",
            sentiment,
            signal,
            verdict,
            "PENDING" # You update this manually later to track accuracy
        ])
    print(f"   📝 Prediction logged to '{filename}'.")

# ---------------------------------------------------------
# 3. CORE ANALYSIS ENGINE
# ---------------------------------------------------------

def analyze_stock(stock_name, ticker_symbol):
    print(f"\n" + "="*70)
    print(f"🔍 ANALYZING: {stock_name.upper()} ({ticker_symbol.upper()})")
    print("="*70)
    
    # --- PHASE 1: NEWS & SENTIMENT ---
    print("📰 FETCHING LATEST NEWS (Last 7 Days)...")
    
    try:
        # Fetch news specifically for India
        google_news = GNews(language='en', country='IN', period='7d', max_results=5)
        news = google_news.get_news(stock_name)
    except Exception as e:
        print(f"   ❌ Connection Error: {e}")
        news = []

    sentiment_score_total = 0
    news_count = 0
    
    if news:
        print(f"   Found {len(news)} relevant articles:\n")
        
        for item in news:
            headline = item['title']
            # Get FinBERT score
            score, label = get_sentiment_score(headline)
            sentiment_score_total += score
            news_count += 1
            
            # Visual formatting
            if label == 'Positive':
                prefix = "🟢 [POS]"
            elif label == 'Negative':
                prefix = "🔴 [NEG]"
            else:
                prefix = "⚪ [NEU]"
            
            # Clean headline print
            clean_head = (headline[:85] + '..') if len(headline) > 85 else headline
            print(f"   {prefix} {clean_head}")
            
        avg_score = sentiment_score_total / news_count
    else:
        print("   ⚠️ No recent news found. Assuming Neutral Sentiment.")
        avg_score = 0

    # Determine Overall Sentiment Status
    if avg_score > 0.15: 
        sentiment_status = "POSITIVE 🟢"
    elif avg_score < -0.15: 
        sentiment_status = "NEGATIVE 🔴"
    else: 
        sentiment_status = "NEUTRAL ⚪"
        
    print(f"\n   🧠 AGGREGATE SENTIMENT SCORE: {avg_score:.2f} ({sentiment_status})")

    # --- PHASE 2: TECHNICAL ANALYSIS ---
    print("\n📈 ANALYZING PRICE ACTION...")
    tech_signal, price, ma50, rsi = get_technical_signal(ticker_symbol)
    
    print(f"   📊 Price: ₹{price:.2f}")
    print(f"   📊 50-MA: ₹{ma50:.2f} (Trend Baseline)")
    print(f"   📊 RSI:   {rsi:.2f} (Strength)")
    
    # --- PHASE 3: FINAL VERDICT ---
    print("\n" + "-"*70)
    print(f"🔮 FINAL PREDICTION FOR {ticker_symbol.upper()}")
    print("-" * 70)
    print(f"   1. NEWS SENTIMENT:  {sentiment_status}")
    print(f"   2. PRICE TREND:     {tech_signal}")
    print("-" * 30)
    
    # Decision Logic
    final_verdict = "HOLD / WAIT"
    
    if "POSITIVE" in sentiment_status and "BULLISH" in tech_signal:
        final_verdict = "🚀 STRONG BUY"
    elif "NEGATIVE" in sentiment_status and "BEARISH" in tech_signal:
        final_verdict = "🔻 STRONG SELL"
    elif "POSITIVE" in sentiment_status and "BEARISH" in tech_signal:
        final_verdict = "⚠️ WATCH (Good News vs. Downtrend)"
    elif "NEGATIVE" in sentiment_status and "BULLISH" in tech_signal:
        final_verdict = "⚠️ CAUTION (Bad News vs. Uptrend)"
    
    print(f"   🎯 RECOMMENDATION: {final_verdict}")
    print("="*70)
    
    # Log the result
    log_prediction(ticker_symbol, sentiment_status, tech_signal, final_verdict, price)
    print("\n")

# ---------------------------------------------------------
# 4. MAIN EXECUTION LOOP
# ---------------------------------------------------------
if __name__ == "__main__":
    while True:
        try:
            print("--------------------------------------------------")
            stock_input = input("Enter Stock Name (e.g. Tata Motors) [or 'q' to quit]: ").strip()
            if stock_input.lower() in ['q', 'exit']:
                print("👋 Exiting...")
                break
            
            if not stock_input:
                continue

            ticker_input = input("Enter Ticker Symbol (e.g. TATAMOTORS.NS): ").strip()
            if not ticker_input:
                print("⚠️ Ticker is required!")
                continue
                
            analyze_stock(stock_input, ticker_input)
            
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break