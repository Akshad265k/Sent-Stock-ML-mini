# # # import yfinance as yf
# # # from gnews import GNews
# # # from transformers import BertTokenizer, BertForSequenceClassification, pipeline
# # # import pandas as pd
# # # import numpy as np
# # # from datetime import datetime, timedelta

# # # # ---------------------------------------------------------
# # # # 1. SETUP THE AI BRAIN (Load FinBERT once)
# # # # ---------------------------------------------------------
# # # print("🧠 Loading AI Sentiment Model (FinBERT)...")
# # # finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
# # # tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
# # # nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer, device=-1)

# # # # Mapping: FinBERT returns labels -> We convert to Score
# # # # Positive=+1, Neutral=0, Negative=-1
# # # score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

# # # # ---------------------------------------------------------
# # # # 2. HELPER FUNCTIONS
# # # # ---------------------------------------------------------
# # # def get_sentiment_score(headline):
# # #     """Returns a numerical score (-1 to 1) for a headline"""
# # #     result = nlp(headline)[0]
# # #     return score_map[result['label']]

# # # def get_technical_signal(ticker):
# # #     """Calculates simple technical indicators (RSI & MA)"""
# # #     stock = yf.Ticker(ticker)
# # #     # Get last 100 days of data
# # #     hist = stock.history(period="3mo")
    
# # #     if len(hist) < 50:
# # #         return "NEUTRAL (Not enough data)"
    
# # #     # 1. Moving Average (50-day)
# # #     ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
# # #     current_price = hist['Close'].iloc[-1]
    
# # #     # 2. RSI (Relative Strength Index) - Simple version
# # #     delta = hist['Close'].diff()
# # #     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
# # #     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
# # #     rs = gain / loss
# # #     rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
# # #     print(f"   📊 Price: ₹{current_price:.2f} | 50-MA: ₹{ma_50:.2f} | RSI: {rsi:.2f}")
    
# # #     if current_price > ma_50 and rsi < 70:
# # #         return "BULLISH (Uptrend)"
# # #     elif current_price < ma_50 and rsi > 30:
# # #         return "BEARISH (Downtrend)"
# # #     else:
# # #         return "NEUTRAL (Sideways)"

# # # # ---------------------------------------------------------
# # # # 3. MAIN PREDICTION LOOP
# # # # ---------------------------------------------------------
# # # def analyze_stock(stock_name, ticker_symbol):
# # #     print(f"\n🔍 ANALYZING: {stock_name} ({ticker_symbol})")
# # #     print("="*50)
    
# # #     # --- STEP A: Get News Sentiment ---
# # #     print("📰 Fetching News...")
# # #     google_news = GNews(language='en', country='IN', period='7d', max_results=10)
# # #     news = google_news.get_news(stock_name)
    
# # #     sentiment_total = 0
# # #     news_count = 0
    
# # #     if news:
# # #         print(f"   Found {len(news)} recent articles.")
# # #         for item in news:
# # #             headline = item['title']
# # #             score = get_sentiment_score(headline)
# # #             sentiment_total += score
# # #             news_count += 1
# # #             # Optional: Print individual headlines
# # #             # print(f"   [{score}] {headline[:60]}...")
            
# # #         avg_sentiment = sentiment_total / news_count
# # #     else:
# # #         print("   ❌ No recent news found.")
# # #         avg_sentiment = 0

# # #     # Interpret Sentiment
# # #     if avg_sentiment > 0.2: sentiment_status = "POSITIVE 🟢"
# # #     elif avg_sentiment < -0.2: sentiment_status = "NEGATIVE 🔴"
# # #     else: sentiment_status = "NEUTRAL ⚪"
    
# # #     print(f"   🧠 AI Sentiment Score: {avg_sentiment:.2f} ({sentiment_status})")

# # #     # --- STEP B: Get Technical Signal ---
# # #     print("\n📈 Analyzing Price Charts...")
# # #     tech_signal = get_technical_signal(ticker_symbol)
    
# # #     # --- STEP C: Final Verdict ---
# # #     print("\n" + "="*50)
# # #     print(f"🔮 FINAL PREDICTION FOR {ticker_symbol}")
# # #     print("="*50)
# # #     print(f"   1. NEWS SENTIMENT:  {sentiment_status}")
# # #     print(f"   2. PRICE TREND:     {tech_signal}")
# # #     print("-" * 30)
    
# # #     # Simple Logic for Final Call
# # #     if "POSITIVE" in sentiment_status and "BULLISH" in tech_signal:
# # #         print("   🚀 RECOMMENDATION: STRONG BUY")
# # #     elif "NEGATIVE" in sentiment_status and "BEARISH" in tech_signal:
# # #         print("   🔻 RECOMMENDATION: STRONG SELL")
# # #     elif "POSITIVE" in sentiment_status and "BEARISH" in tech_signal:
# # #         print("   ⚠️ CONFLICT: News is Good, but Trend is Down (Watch for reversal)")
# # #     else:
# # #         print("   ✋ RECOMMENDATION: HOLD / WAIT")
# # #     print("="*50 + "\n")

# # # # ---------------------------------------------------------
# # # # 4. RUN IT
# # # # ---------------------------------------------------------
# # # # Try with Indian Stocks
# # # analyze_stock("Tata Motors", "TATAMOTORS.NS")
# # # analyze_stock("Infosys", "INFY.NS")
# # # # Run this specific command in your terminal
# # # analyze_stock("Adani Enterprises", "ADANIENT.NS")
# # # analyze_stock("Reliance Industries", "RELIANCE.NS")
# # # analyze_stock("Bata India", "BATAINDIA.NS")

# # import yfinance as yf
# # from gnews import GNews
# # from transformers import BertTokenizer, BertForSequenceClassification, pipeline
# # import pandas as pd
# # import numpy as np

# # # ---------------------------------------------------------
# # # 1. SETUP THE AI BRAIN
# # # ---------------------------------------------------------
# # print("🧠 Loading AI Sentiment Model (FinBERT)...")
# # # FinBERT is specific for financial language (e.g., "Cost reduction" = Positive)
# # finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
# # tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
# # nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer, device=-1)

# # # Mapping labels to a numerical score
# # score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

# # # ---------------------------------------------------------
# # # 2. HELPER FUNCTIONS
# # # ---------------------------------------------------------
# # def get_sentiment_score(headline):
# #     """
# #     Returns a tuple: (Numerical Score, Label)
# #     e.g., (1, 'Positive')
# #     """
# #     # FinBERT returns a list of dicts: [{'label': 'Positive', 'score': 0.99}]
# #     result = nlp(headline)[0]
# #     label = result['label']
# #     score = score_map[label]
# #     return score, label

# # def get_technical_signal(ticker):
# #     """Calculates simple technical indicators (RSI & MA)"""
# #     try:
# #         stock = yf.Ticker(ticker)
# #         # Get last 3 months of data to calculate Moving Averages
# #         hist = stock.history(period="3mo")
        
# #         if len(hist) < 50:
# #             return "NEUTRAL (Not enough data)", 0, 0
        
# #         # 1. Moving Average (50-day)
# #         ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
# #         current_price = hist['Close'].iloc[-1]
        
# #         # 2. RSI (Relative Strength Index)
# #         delta = hist['Close'].diff()
# #         gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
# #         loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
# #         rs = gain / loss
# #         rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
# #         print(f"\n   📊 MARKET DATA:")
# #         print(f"      Current Price: ₹{current_price:.2f}")
# #         print(f"      50-Day Avg:    ₹{ma_50:.2f}")
# #         print(f"      RSI Strength:  {rsi:.2f}")
        
# #         # Logic for Signal
# #         if current_price > ma_50 and rsi < 70:
# #             return "BULLISH (Uptrend)", current_price, rsi
# #         elif current_price < ma_50 and rsi > 30:
# #             return "BEARISH (Downtrend)", current_price, rsi
# #         elif rsi <= 30:
# #             return "NEUTRAL (Oversold - Risky to Sell)", current_price, rsi
# #         else:
# #             return "NEUTRAL (Sideways)", current_price, rsi
            
# #     except Exception as e:
# #         print(f"   ❌ Error fetching price data: {e}")
# #         return "UNKNOWN", 0, 0

# # # ---------------------------------------------------------
# # # 3. MAIN PREDICTION LOOP
# # # ---------------------------------------------------------
# # def analyze_stock(stock_name, ticker_symbol):
# #     print(f"\n" + "="*60)
# #     print(f"🔍 ANALYZING: {stock_name} ({ticker_symbol})")
# #     print("="*60)
    
# #     # --- STEP A: Get News Sentiment ---
# #     print("📰 FETCHING & ANALYZING NEWS...")
# #     try:
# #         google_news = GNews(language='en', country='IN', period='7d', max_results=5)
# #         news = google_news.get_news(stock_name)
# #     except Exception as e:
# #         print(f"   ❌ Error connecting to Google News: {e}")
# #         news = []
    
# #     sentiment_total = 0
# #     news_count = 0
    
# #     if news:
# #         print(f"   Found {len(news)} recent articles:\n")
        
# #         for item in news:
# #             headline = item['title']
# #             score, label = get_sentiment_score(headline)
# #             sentiment_total += score
# #             news_count += 1
            
# #             # VISUALIZATON: Print individual headlines with scores
# #             # Using simple ASCII indicators for Positive/Negative
# #             if score == 1:
# #                 prefix = "🟢 [POS]"
# #             elif score == -1:
# #                 prefix = "🔴 [NEG]"
# #             else:
# #                 prefix = "⚪ [NEU]"
                
# #             # Truncate headline for cleaner output
# #             clean_headline = (headline[:75] + '..') if len(headline) > 75 else headline
# #             print(f"   {prefix} {clean_headline}")

# #         avg_sentiment = sentiment_total / news_count
# #     else:
# #         print("   ❌ No recent news found.")
# #         avg_sentiment = 0

# #     # Interpret Average Sentiment
# #     if avg_sentiment > 0.2: sentiment_status = "POSITIVE 🟢"
# #     elif avg_sentiment < -0.2: sentiment_status = "NEGATIVE 🔴"
# #     else: sentiment_status = "NEUTRAL ⚪"
    
# #     print(f"\n   🧠 AVG AI SENTIMENT SCORE: {avg_sentiment:.2f} ({sentiment_status})")

# #     # --- STEP B: Get Technical Signal ---
# #     print("\n📈 ANALYZING PRICE CHARTS...")
# #     tech_signal, price, rsi = get_technical_signal(ticker_symbol)
    
# #     # --- STEP C: Final Verdict ---
# #     print("\n" + "-"*60)
# #     print(f"🔮 FINAL PREDICTION FOR {ticker_symbol}")
# #     print("-" * 60)
# #     print(f"   1. NEWS SENTIMENT:  {sentiment_status}")
# #     print(f"   2. PRICE TREND:     {tech_signal}")
# #     print("-" * 30)
    
# #     # Logic for Final Call
# #     if "POSITIVE" in sentiment_status and "BULLISH" in tech_signal:
# #         print("   🚀 RECOMMENDATION: STRONG BUY")
# #     elif "NEGATIVE" in sentiment_status and "BEARISH" in tech_signal:
# #         print("   🔻 RECOMMENDATION: STRONG SELL")
# #     elif "POSITIVE" in sentiment_status and "BEARISH" in tech_signal:
# #         print("   ⚠️ CONFLICT: News is Good, but Trend is Down (Watch for Reversal)")
# #     elif "NEGATIVE" in sentiment_status and "BULLISH" in tech_signal:
# #         print("   ⚠️ CONFLICT: Trend is Up, but News is Bad (Possible Top/Exit)")
# #     else:
# #         print("   ✋ RECOMMENDATION: HOLD / WAIT")
# #     print("="*60 + "\n")

# # # ---------------------------------------------------------
# # # 4. RUNNER
# # # ---------------------------------------------------------
# # if __name__ == "__main__":
# #     while True:
# #         user_input = input("Enter Stock Name (e.g., Tata Motors, Bata India) or 'q' to quit: ")
# #         if user_input.lower() == 'q':
# #             break
        
# #         # Simple helper to guess the ticker (You can improve this later)
# #         # This assumes the user types the exact Yahoo ticker or we default to .NS
# #         ticker_input = input("Enter Ticker Symbol (e.g. TATAMOTORS.NS, BATAINDIA.NS): ")
        
# #         analyze_stock(user_input, ticker_input)



# # #5
# # import csv
# # import os
# # from datetime import datetime

# # def log_prediction(ticker, sentiment, signal, prediction, current_price):
# #     """Saves the prediction to a CSV file to check accuracy later"""
# #     file_exists = os.path.isfile('trading_journal.csv')
    
# #     with open('trading_journal.csv', 'a', newline='') as f:
# #         writer = csv.writer(f)
# #         # Write header if new file
# #         if not file_exists:
# #             writer.writerow(['Date', 'Ticker', 'Price', 'Sentiment', 'Signal', 'Prediction', 'Outcome (Check Later)'])
            
# #         writer.writerow([
# #             datetime.now().strftime("%Y-%m-%d"), 
# #             ticker, 
# #             current_price, 
# #             sentiment, 
# #             signal, 
# #             prediction, 
# #             "PENDING"
# #         ])
# #     print(f"   📝 Prediction logged to 'trading_journal.csv'. Check back in 1 week!")


# import yfinance as yf
# from gnews import GNews
# from transformers import BertTokenizer, BertForSequenceClassification, pipeline
# import pandas as pd
# import numpy as np
# import csv
# import os
# from datetime import datetime

# # ---------------------------------------------------------
# # 1. GLOBAL SETUP (Load AI Model Once)
# # ---------------------------------------------------------
# print("\n🧠 INITIALIZING SYSTEM...")
# print("   Loading FinBERT Financial Sentiment Model... (This may take a moment)")

# # Load FinBERT (Pre-trained on financial text)
# try:
#     finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
#     tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
#     # device=-1 uses CPU (change to 0 if you have a GPU set up with CUDA)
#     nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer, device=-1)
#     print("   ✅ Model Loaded Successfully.")
# except Exception as e:
#     print(f"   ❌ Error loading model: {e}")
#     exit()

# # Score Mapping: Positive=1, Neutral=0, Negative=-1
# score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

# # ---------------------------------------------------------
# # 2. HELPER FUNCTIONS
# # ---------------------------------------------------------

# def get_sentiment_score(headline):
#     """
#     Analyzes a single headline and returns the numerical score and label.
#     """
#     # Truncate to 512 tokens to match BERT limits
#     results = nlp(headline[:512]) 
#     result = results[0]
#     label = result['label']
#     score = score_map[label]
#     return score, label

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

# def log_prediction(ticker, sentiment, signal, verdict, price):
#     """
#     Logs the prediction to a CSV file for accuracy tracking.
#     """
#     filename = 'trading_journal.csv'
#     file_exists = os.path.isfile(filename)
    
#     with open(filename, 'a', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         # Write header if file is new
#         if not file_exists:
#             writer.writerow(['Date', 'Ticker', 'Price', 'Sentiment_Status', 'Tech_Signal', 'Final_Verdict', 'Result'])
            
#         writer.writerow([
#             datetime.now().strftime("%Y-%m-%d %H:%M"),
#             ticker,
#             f"{price:.2f}",
#             sentiment,
#             signal,
#             verdict,
#             "PENDING" # You update this manually later to track accuracy
#         ])
#     print(f"   📝 Prediction logged to '{filename}'.")

# # ---------------------------------------------------------
# # 3. CORE ANALYSIS ENGINE
# # ---------------------------------------------------------

# def analyze_stock(stock_name, ticker_symbol):
#     print(f"\n" + "="*70)
#     print(f"🔍 ANALYZING: {stock_name.upper()} ({ticker_symbol.upper()})")
#     print("="*70)
    
#     # --- PHASE 1: NEWS & SENTIMENT ---
#     print("📰 FETCHING LATEST NEWS (Last 7 Days)...")
    
#     try:
#         # Fetch news specifically for India
#         google_news = GNews(language='en', country='IN', period='7d', max_results=5)
#         news = google_news.get_news(stock_name)
#     except Exception as e:
#         print(f"   ❌ Connection Error: {e}")
#         news = []

#     sentiment_score_total = 0
#     news_count = 0
    
#     if news:
#         print(f"   Found {len(news)} relevant articles:\n")
        
#         for item in news:
#             headline = item['title']
#             # Get FinBERT score
#             score, label = get_sentiment_score(headline)
#             sentiment_score_total += score
#             news_count += 1
            
#             # Visual formatting
#             if label == 'Positive':
#                 prefix = "🟢 [POS]"
#             elif label == 'Negative':
#                 prefix = "🔴 [NEG]"
#             else:
#                 prefix = "⚪ [NEU]"
            
#             # Clean headline print
#             clean_head = (headline[:85] + '..') if len(headline) > 85 else headline
#             print(f"   {prefix} {clean_head}")
            
#         avg_score = sentiment_score_total / news_count
#     else:
#         print("   ⚠️ No recent news found. Assuming Neutral Sentiment.")
#         avg_score = 0

#     # Determine Overall Sentiment Status
#     if avg_score > 0.15: 
#         sentiment_status = "POSITIVE 🟢"
#     elif avg_score < -0.15: 
#         sentiment_status = "NEGATIVE 🔴"
#     else: 
#         sentiment_status = "NEUTRAL ⚪"
        
#     print(f"\n   🧠 AGGREGATE SENTIMENT SCORE: {avg_score:.2f} ({sentiment_status})")

#     # --- PHASE 2: TECHNICAL ANALYSIS ---
#     print("\n📈 ANALYZING PRICE ACTION...")
#     tech_signal, price, ma50, rsi = get_technical_signal(ticker_symbol)
    
#     print(f"   📊 Price: ₹{price:.2f}")
#     print(f"   📊 50-MA: ₹{ma50:.2f} (Trend Baseline)")
#     print(f"   📊 RSI:   {rsi:.2f} (Strength)")
    
#     # --- PHASE 3: FINAL VERDICT ---
#     print("\n" + "-"*70)
#     print(f"🔮 FINAL PREDICTION FOR {ticker_symbol.upper()}")
#     print("-" * 70)
#     print(f"   1. NEWS SENTIMENT:  {sentiment_status}")
#     print(f"   2. PRICE TREND:     {tech_signal}")
#     print("-" * 30)
    
#     # Decision Logic
#     final_verdict = "HOLD / WAIT"
    
#     if "POSITIVE" in sentiment_status and "BULLISH" in tech_signal:
#         final_verdict = "🚀 STRONG BUY"
#     elif "NEGATIVE" in sentiment_status and "BEARISH" in tech_signal:
#         final_verdict = "🔻 STRONG SELL"
#     elif "POSITIVE" in sentiment_status and "BEARISH" in tech_signal:
#         final_verdict = "⚠️ WATCH (Good News vs. Downtrend)"
#     elif "NEGATIVE" in sentiment_status and "BULLISH" in tech_signal:
#         final_verdict = "⚠️ CAUTION (Bad News vs. Uptrend)"
    
#     print(f"   🎯 RECOMMENDATION: {final_verdict}")
#     print("="*70)
    
#     # Log the result
#     log_prediction(ticker_symbol, sentiment_status, tech_signal, final_verdict, price)
#     print("\n")

# # ---------------------------------------------------------
# # 4. MAIN EXECUTION LOOP
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     while True:
#         try:
#             print("--------------------------------------------------")
#             stock_input = input("Enter Stock Name (e.g. Tata Motors) [or 'q' to quit]: ").strip()
#             if stock_input.lower() in ['q', 'exit']:
#                 print("👋 Exiting...")
#                 break
            
#             if not stock_input:
#                 continue

#             ticker_input = input("Enter Ticker Symbol (e.g. TATAMOTORS.NS): ").strip()
#             if not ticker_input:
#                 print("⚠️ Ticker is required!")
#                 continue
                
#             analyze_stock(stock_input, ticker_input)
            
#         except KeyboardInterrupt:
#             print("\n👋 Exiting...")
#             break


# finbert_tf_full_pipeline.py
import os

# ---------------------------
# CRITICAL FIX FOR TF 2.16+ / KERAS 3
# ---------------------------
# This forces TensorFlow to use the legacy Keras 2 behavior 
# which is required for Hugging Face Transformers compatibility.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from gnews import GNews

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from transformers import (
    BertTokenizerFast,
    TFBertModel,
    logging as hf_logging
)

# ... (rest of your code remains exactly the same)
hf_logging.set_verbosity_error()  # reduce transformers logs

# ---------------------------
# CONFIG
# ---------------------------
PRETRAINED_NAME = "yiyanghkust/finbert-tone"
FINETUNED_SAVE_DIR = "./finbert_tf_finetuned"
os.makedirs(FINETUNED_SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 128
LR = 2e-5
LABEL_MAP = {0: "Neutral", 1: "Positive", 2: "Negative"}
SCORE_MAP = {"Positive": 1, "Neutral": 0, "Negative": -1}
DEVICE = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

print(f"\nRunning on: {DEVICE}")

# ---------------------------
# TOKENIZER + BASE MODEL
# ---------------------------
print("Loading tokenizer and base BERT (this may take a moment)...")
tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_NAME)

# Use from_pt=True if TF weights not available for that model; TFBertModel will attempt either.
try:
    bert_encoder = TFBertModel.from_pretrained(PRETRAINED_NAME, from_pt=True)  # safe fallback
except Exception:
    bert_encoder = TFBertModel.from_pretrained(PRETRAINED_NAME)

# ---------------------------
# Keras Model wrapper
# ---------------------------
class BertWrapper(layers.Layer):
    def __init__(self, bert_model, **kwargs):
        super().__init__(**kwargs)
        self.bert = bert_model

    def call(self, inputs):
        ids, mask = inputs
        outputs = self.bert(ids, attention_mask=mask, training=False)
        return outputs.pooler_output  # ALWAYS (batch, 768)

    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]  # same batch size
        hidden = self.bert.config.hidden_size
        return (batch, hidden)

def build_sentiment_model(bert_model, max_len=MAX_LEN, dropout_rate=0.3):
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    pooled = BertWrapper(bert_model)([input_ids, attention_mask])

    x = layers.Dropout(dropout_rate)(pooled)
    x = layers.Dense(bert_model.config.hidden_size // 2, activation="relu")(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    logits = layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs=[input_ids, attention_mask], outputs=logits)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model



print("Building Keras model...")
sent_model = build_sentiment_model(bert_encoder)
sent_model.summary()

# ---------------------------
# DATA PREP / TF.DATA
# ---------------------------
def df_to_tf_dataset(df, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE, shuffle=True):
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    enc = tokenizer(
        texts,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="np"
    )

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"]
        },
        np.array(labels)
    ))
    if shuffle:
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ---------------------------
# FINETUNE ENTRYPOINT
# ---------------------------
def finetune_from_csv_tf(csv_path, save_dir=FINETUNED_SAVE_DIR,
                         epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    CSV must contain columns: text,label
    label values: 0=Neutral,1=Positive,2=Negative
    """
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    # simple stratified split-ish: shuffle then split
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_frac = 0.9
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    train_ds = df_to_tf_dataset(train_df, tokenizer, batch_size=batch_size)
    val_ds = df_to_tf_dataset(val_df, tokenizer, batch_size=batch_size, shuffle=False)

    # Build a fresh model (fresh head) to train
    model = build_sentiment_model(bert_encoder)

    # Fit
    print(f"Starting training: {len(train_df)} train / {len(val_df)} val examples")
    history = 0
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # Save whole Keras model (incl. BERT weights)
    model_path = os.path.join(save_dir, "sentiment_keras_model")
    model.save(model_path, include_optimizer=False)
    # save tokenizer
    tokenizer.save_pretrained(save_dir)
    print("Saved finetuned model to:", model_path)
    return model, history

# ---------------------------
# INFERENCE UTILITIES
# ---------------------------
def load_finetuned_tf_model(model_dir=os.path.join(FINETUNED_SAVE_DIR, "sentiment_keras_model")):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No saved model at {model_dir}. Train first or point to correct path.")
    model = keras.models.load_model(model_dir, compile=False)
    return model

def predict_sentiment_tf(model, texts, tokenizer=tokenizer, max_len=MAX_LEN):
    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True

    enc = tokenizer(
        texts,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="np"
    )
    preds = model.predict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}, verbose=0)
    classes = np.argmax(preds, axis=1)
    out = []
    for c in classes:
        label = LABEL_MAP[int(c)]
        score = SCORE_MAP[label]
        out.append((score, label))
    return out[0] if single else out

# ---------------------------
# Robust Technical Analysis (ALWAYS returns a trend)
# ---------------------------
def get_technical_signal_robust(ticker):
    """
    Robust technical analysis that ALWAYS produces a price trend:
    - Uses MA50, fallback to MA20, fallback to EMA10
    - Uses RSI14, fallback to 5-day momentum
    - Works even with limited Yahoo data
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")

        if hist is None or len(hist) == 0:
            return "UNKNOWN (No Market Data Found)", 0.0, 0.0, 0.0

        hist = hist[["Close"]].dropna()
        hist = hist.sort_index()
        hist["Close"] = hist["Close"].ffill().bfill()

        price = float(hist["Close"].iloc[-1])

        ma_50 = hist["Close"].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None
        ma_20 = hist["Close"].rolling(20).mean().iloc[-1] if len(hist) >= 20 else None
        ema_10 = hist["Close"].ewm(span=10).mean().iloc[-1]

        # choose best baseline
        if ma_50 is not None and not np.isnan(ma_50):
            baseline = float(ma_50)
            baseline_type = "MA50"
        elif ma_20 is not None and not np.isnan(ma_20):
            baseline = float(ma_20)
            baseline_type = "MA20"
        else:
            baseline = float(ema_10)
            baseline_type = "EMA10"

        # RSI or fallback momentum
        if len(hist) >= 15:
            delta = hist["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])
            if np.isnan(rsi):
                # fallback momentum
                lookback = min(5, len(hist) - 1)
                rsi = float(((price - hist["Close"].iloc[-lookback-1]) / hist["Close"].iloc[-lookback-1]) * 100)
        else:
            # fallback momentum
            lookback = min(5, len(hist)-1)
            if lookback <= 0:
                rsi = 50.0
            else:
                rsi = float(((price - hist["Close"].iloc[-lookback-1]) / hist["Close"].iloc[-lookback-1]) * 100)

        # Determine trend
        trend = "NEUTRAL"
        if price > baseline:
            if rsi < 70:
                trend = f"BULLISH (Above {baseline_type})"
            else:
                trend = f"NEUTRAL (Overbought vs {baseline_type})"
        elif price < baseline:
            if rsi > 30:
                trend = f"BEARISH (Below {baseline_type})"
            else:
                trend = f"NEUTRAL (Oversold vs {baseline_type})"

        return trend, price, baseline, rsi

    except Exception as e:
        return f"ERROR ({e})", 0.0, 0.0, 0.0

# ---------------------------
# Logging utility
# ---------------------------
def log_prediction(ticker, sentiment, signal, verdict, price):
    filename = 'trading_journal.csv'
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Ticker', 'Price', 'Sentiment_Status', 'Tech_Signal', 'Final_Verdict', 'Result'])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            ticker,
            f"{price:.2f}",
            sentiment,
            signal,
            verdict,
            "PENDING"
        ])
    print(f"   📝 Prediction logged to '{filename}'.")

# ---------------------------
# Core analysis engine (uses TF model for sentiment)
# ---------------------------
def analyze_stock(stock_name, ticker_symbol, sentiment_model):
    print(f"\n" + "="*70)
    print(f"🔍 ANALYZING: {stock_name.upper()} ({ticker_symbol.upper()})")
    print("="*70)

    # News & sentiment
    print("📰 FETCHING LATEST NEWS (Last 7 Days)...")
    try:
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
            headline = item.get('title', '')
            score, label = predict_sentiment_tf(sentiment_model, headline)
            sentiment_score_total += score
            news_count += 1

            prefix = "🟢 [POS]" if label == "Positive" else ("🔴 [NEG]" if label == "Negative" else "⚪ [NEU]")
            clean_head = (headline[:85] + '..') if len(headline) > 85 else headline
            print(f"   {prefix} {clean_head}")
        avg_score = sentiment_score_total / news_count
    else:
        print("   ⚠ No recent news found. Assuming Neutral Sentiment.")
        avg_score = 0

    if avg_score > 0.15:
        sentiment_status = "POSITIVE 🟢"
    elif avg_score < -0.15:
        sentiment_status = "NEGATIVE 🔴"
    else:
        sentiment_status = "NEUTRAL ⚪"

    print(f"\n   🧠 AGGREGATE SENTIMENT SCORE: {avg_score:.2f} ({sentiment_status})")

    # Technical analysis
    print("\n📈 ANALYZING PRICE ACTION...")
    tech_signal, price, baseline, rsi = get_technical_signal_robust(ticker_symbol)

    print(f"   📊 Price: ₹{price:.2f}")
    print(f"   📊 Baseline: ₹{baseline:.2f}")
    print(f"   📊 RSI:   {rsi:.2f}")

    # Final verdict
    print("\n" + "-"*70)
    print(f"🔮 FINAL PREDICTION FOR {ticker_symbol.upper()}")
    print("-" * 70)
    print(f"   1. NEWS SENTIMENT:  {sentiment_status}")
    print(f"   2. PRICE TREND:     {tech_signal}")
    print("-" * 30)

    final_verdict = "HOLD / WAIT"
    if "POSITIVE" in sentiment_status and "BULLISH" in tech_signal:
        final_verdict = "🚀 STRONG BUY"
    elif "NEGATIVE" in sentiment_status and "BEARISH" in tech_signal:
        final_verdict = "🔻 STRONG SELL"
    elif "POSITIVE" in sentiment_status and "BEARISH" in tech_signal:
        final_verdict = "⚠ WATCH (Good News vs. Downtrend)"
    elif "NEGATIVE" in sentiment_status and "BULLISH" in tech_signal:
        final_verdict = "⚠ CAUTION (Bad News vs. Uptrend)"

    print(f"   🎯 RECOMMENDATION: {final_verdict}")
    print("="*70)

    log_prediction(ticker_symbol, sentiment_status, tech_signal, final_verdict, price)
    print("\n")

def prepare_and_finetune(raw_csv_path="all-data.csv"):
    """
    Converts 'all-data.csv' (No header, string labels) 
    to 'finbert_training_data.csv' (Header, int labels)
    and then runs finetuning.
    """
    if not os.path.exists(raw_csv_path):
        print(f"❌ File '{raw_csv_path}' not found. Please place it in the same folder.")
        return

    print(f"🔄 Processing {raw_csv_path}...")
    
    # 1. Read dataset (handle encoding)
    try:
        # header=None because the file starts directly with data
        # names=['sentiment', 'text'] assigns column names
        df = pd.read_csv(raw_csv_path, header=None, names=['sentiment', 'text'], encoding='latin-1')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Map string sentiment to integers matching your LABEL_MAP
    # Your config: {0: "Neutral", 1: "Positive", 2: "Negative"}
    label_mapping = {
        "neutral": 0,
        "positive": 1,
        "negative": 2
    }
    
    # Filter rows to ensure only valid labels exist
    df = df[df['sentiment'].isin(label_mapping.keys())].copy()
    
    # Apply mapping
    df['label'] = df['sentiment'].map(label_mapping)
    
    # 3. Save in the format finetune_from_csv_tf expects (columns: text, label)
    ready_file = "finbert_training_data.csv"
    df[['text', 'label']].to_csv(ready_file, index=False)
    
    print(f"✅ Converted dataset saved to '{ready_file}' ({len(df)} rows).")
    print("🚀 Starting FinBERT Fine-tuning...")

    # 4. Call your existing training function
    finetune_from_csv_tf(ready_file, epochs=3)
  # ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("\nFinBERT (TF) sentiment + robust technical analysis script ready.")

    # --- AUTO-TRAINING LOGIC ---
    # If "all-data.csv" exists and we haven't trained yet, ask to train.
    if os.path.exists("all-data.csv"):
        choice = input("Found 'all-data.csv'. Do you want to finetune the model now? (y/n): ").strip().lower()
        if choice == 'y':
            prepare_and_finetune("all-data.csv")
            # After training, reload the NEW model
            try:
                sentiment_model = load_finetuned_tf_model()
                print("✅ Loaded newly finetuned model.")
            except:
                print("⚠ Could not load new model. Using base model.")
                sentiment_model = sent_model
        else:
            print("Skipping training.")
            # Load existing if available
            try:
                sentiment_model = load_finetuned_tf_model()
                print("Loaded existing finetuned TF model.")
            except:
                print("No finetuned model found. Using base model.")
                sentiment_model = sent_model
    else:
        # Standard loading logic if no dataset is found
        try:
            sentiment_model = load_finetuned_tf_model()
            print("Loaded finetuned TF model.")
        except:
            print("No finetuned model found. Using fresh model (untrained head).")
            sentiment_model = sent_model 

    # --- INFERENCE LOOP ---
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
                print("⚠ Ticker is required!")
                continue

            analyze_stock(stock_input, ticker_input, sentiment_model)

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break