import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import yfinance as yf
from gnews import GNews


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizerFast

# 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MAX_LEN = 128
MODEL_PATH = r"./best_sentiment_model_1.keras"
TOKENIZER_PATH = r"./finbert_tf_finetuned"


print(os.path.exists(MODEL_PATH))
print(os.path.isdir(TOKENIZER_PATH))


# Labels Config
LABEL_MAP = {0: "Neutral", 1: "Positive", 2: "Negative"}
SCORE_MAP = {"Positive": 1, "Neutral": 0, "Negative": -1}

# --- Define the Wrapper Class (Required for loading) ---
# We must re-define this class so Keras knows how to reconstruct the model
@keras.utils.register_keras_serializable()
class BertWrapper(layers.Layer):
    def __init__(self, bert_model=None, **kwargs):
        super().__init__(**kwargs)
        self.bert = bert_model

    def call(self, inputs):
        ids, mask = inputs
        # training=False is crucial for inference
        outputs = self.bert(ids, attention_mask=mask, training=False)
        return outputs.pooler_output

    def get_config(self):
        config = super().get_config()
        return config

print("🧠 Loading Finetuned FinBERT (TensorFlow)...")

# Global variables
model = None
tokenizer = None

try:
    # 1. Load Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    
    # 2. Load Model
    # We pass custom_objects so Keras knows what 'BertWrapper' is
    try:
        model = keras.models.load_model(MODEL_PATH, custom_objects={"BertWrapper": BertWrapper}, compile=False)
    except TypeError:
        # Fallback if standard load fails
        print("⚠️ Standard load failed, attempting safe load...")
        model = tf.saved_model.load(MODEL_PATH)

    print("✅ FinBERT (TF) Loaded Successfully")
except Exception as e:
    print(f"❌ Critical Error Loading Model: {e}")
    print("Ensure you have run the training script and folders exist.")


# ---------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------
def get_sentiment(text: str):
    """
    Runs inference using the TensorFlow model.
    Returns: (score, label)
    """
    if not model or not tokenizer:
        return 0, "Neutral"

    try:
        # 1. Tokenize
        enc = tokenizer(
            [text],  # Wrap in list
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="np"
        )

        # 2. Predict
        if hasattr(model, 'predict'):
            # Keras Model
            probs = model.predict(
                {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
                verbose=0
            )
        else:
            # TF SavedModel (fallback)
            probs = model(
                {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
            )

        # 3. Decode
        pred_idx = np.argmax(probs, axis=1)[0]
        label = LABEL_MAP[pred_idx]
        score = SCORE_MAP[label]
        
        return score, label

    except Exception as e:
        print(f"Sentiment Error: {e}")
        return 0, "Neutral"


def analyze_technicals(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")

        if len(hist) < 50:
            return None

        current_price = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2])
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100

        prices = [float(p) for p in hist["Close"].tail(30).tolist()]

        ma50 = float(hist["Close"].rolling(50).mean().iloc[-1])

        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        
        if loss.iloc[-1] == 0:
            rsi = 100.0
        else:
            rs = gain / loss
            rsi = float(100 - (100 / (1 + rs)).iloc[-1])

        # Technical Signal
        if current_price > ma50:
            trend = "Uptrend"
            signal = "BUY" if rsi < 70 else "NEUTRAL"
        else:
            trend = "Downtrend"
            signal = "SELL" if rsi > 30 else "NEUTRAL"

        return {
            "current_price": current_price,
            "change": change,
            "change_percent": change_percent,
            "rsi": rsi,
            "ma50": ma50,
            "trend": trend,
            "signal": signal,
            "prices": prices
        }
    except Exception as e:
        print("❌ Technical error:", e)
        return None


# ---------------------------------------------------------
# 4. DATA MODELS
# ---------------------------------------------------------
class StockRequest(BaseModel):
    ticker: str

class Holding(BaseModel):
    ticker: str
    quantity: float
    buyPrice: float

class PortfolioRequest(BaseModel):
    holdings: List[Holding]


# ---------------------------------------------------------
# 5. API — STOCK PREDICTION
# ---------------------------------------------------------
@app.post("/api/predict")
async def predict_stock(request: StockRequest):
    ticker = request.ticker.upper().strip()

    # 1. Technical Analysis
    tech = analyze_technicals(ticker)
    if not tech and "." not in ticker:
        ticker = ticker + ".NS"
        tech = analyze_technicals(ticker)

    if not tech:
        raise HTTPException(
            status_code=404, detail=f"No stock data for {request.ticker}")

    # 2. News & Sentiment
    clean = ticker.replace(".NS", "").replace(".BO", "")
    region = "IN" if ticker.endswith((".NS", ".BO")) else "US"

    try:
        g = GNews(language='en', country=region, period='7d', max_results=5)
        news_items = g.get_news(clean)
    except:
        news_items = []

    news_list = []
    total_sent_score = 0

    if news_items:
        print(f"Analyzing {len(news_items)} articles for {clean}...")
        for n in news_items:
            title = n.get("title", "")
            # --- CALL NEW SENTIMENT FUNCTION ---
            scr, lbl = get_sentiment(title)
            total_sent_score += scr

            news_list.append({
                "title": title,
                "source": n.get("publisher", {}).get("title", "Unknown"),
                "sentiment": lbl,
                "url": n.get("url", "#")
            })

        avg_score = total_sent_score / len(news_items)
    else:
        avg_score = 0

    # Determine Aggregate Label
    if avg_score > 0.15:
        sentiment_label = "Positive"
    elif avg_score < -0.15:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # 3. AI Verdict Logic
    tech_sig = tech["signal"]
    
    # Simple Heuristic
    if tech_sig == "BUY" and sentiment_label == "Positive":
        ai_signal = "BUY"
        tp = tech["current_price"] * 1.05
        conf = 0.85
    elif tech_sig == "SELL" and sentiment_label == "Negative":
        ai_signal = "SELL"
        tp = tech["current_price"] * 0.95
        conf = 0.85
    elif tech_sig == "BUY" and sentiment_label == "Negative":
        ai_signal = "HOLD (Caution)"
        tp = tech["current_price"] * 1.01
        conf = 0.50
    elif tech_sig == "SELL" and sentiment_label == "Positive":
        ai_signal = "HOLD (Watch)"
        tp = tech["current_price"] * 0.99
        conf = 0.50
    else:
        ai_signal = "HOLD"
        tp = tech["current_price"]
        conf = 0.60

    return {
        "ticker": ticker,
        "name": clean,
        "currentPrice": round(tech["current_price"], 2),
        "change": round(tech["change"], 2),
        "changePercent": round(tech["change_percent"], 2),
        "prices": tech["prices"],

        "prediction": {
            "targetPrice": round(tp, 2),
            "confidence": conf,
            "timeframe": "7 Days",
            "signal": ai_signal
        },

        "sentiment": {
            "score": round(avg_score, 2),
            "label": sentiment_label
        },

        "news": news_list
    }


# ---------------------------------------------------------
# 6. API — PORTFOLIO ANALYSIS
# ---------------------------------------------------------
@app.post("/api/portfolio/analyze")
async def analyze_portfolio(request: PortfolioRequest):
    if not request.holdings:
        raise HTTPException(status_code=400, detail="No holdings given")

    results = []
    total_invested = 0
    total_value = 0

    for h in request.holdings:
        try:
            req = StockRequest(ticker=h.ticker)
            stock = await predict_stock(req)
        except Exception as e:
            print(f"Skipping {h.ticker}: {e}")
            continue

        invested = h.buyPrice * h.quantity
        value = stock["currentPrice"] * h.quantity
        pnl = value - invested
        pnl_percent = (pnl / invested * 100) if invested else 0

        total_invested += invested
        total_value += value

        results.append({
            "ticker": stock["ticker"],
            "quantity": h.quantity,
            "buyPrice": h.buyPrice,
            "currentPrice": stock["currentPrice"],
            "invested": round(invested, 2),
            "value": round(value, 2),
            "pnl": round(pnl, 2),
            "pnlPercent": round(pnl_percent, 2),
            "prediction": stock["prediction"],
            "sentiment": stock["sentiment"],
        })

    if not results:
        raise HTTPException(status_code=404, detail="No valid holdings found")

    total_pnl = total_value - total_invested

    overview = {
        "totalInvested": round(total_invested, 2),
        "currentValue": round(total_value, 2),
        "totalPnL": round(total_pnl, 2),
        "totalPnLPercent": round(total_pnl / total_invested * 100, 2) if total_invested else 0
    }

    # Calculate weights
    for r in results:
        r["weight"] = round((r["value"] / total_value * 100) if total_value else 0, 2)

    return {"overview": overview, "stocks": results}


# ---------------------------------------------------------
# 7. RUN SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


