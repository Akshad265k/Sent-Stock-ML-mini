
# # from fastapi import FastAPI, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel
# # from typing import List
# # import yfinance as yf
# # from gnews import GNews
# # from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# # # ---------------------------------------------------------
# # # 1. SETUP API & AI MODELS
# # # ---------------------------------------------------------
# # app = FastAPI()

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # print("🧠 Loading FinBERT...")
# # try:
# #     finbert = BertForSequenceClassification.from_pretrained(
# #         'yiyanghkust/finbert-tone', num_labels=3)
# #     tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
# #     nlp = pipeline("sentiment-analysis",
# #                    model=finbert, tokenizer=tokenizer, device=-1)
# #     score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
# #     print("✅ FinBERT Loaded")
# # except Exception as e:
# #     print("❌ FinBERT Load Failed:", e)


# # # ---------------------------------------------------------
# # # 2. HELPERS
# # # ---------------------------------------------------------
# # def get_sentiment(text: str):
# #     try:
# #         res = nlp(text[:512])[0]
# #         return score_map[res["label"]], res["label"]
# #     except:
# #         return 0, "Neutral"


# # def analyze_technicals(ticker: str):
# #     try:
# #         stock = yf.Ticker(ticker)
# #         hist = stock.history(period="6mo")

# #         if len(hist) < 50:
# #             return None

# #         current_price = float(hist["Close"].iloc[-1])
# #         prev_close = float(hist["Close"].iloc[-2])
# #         change = current_price - prev_close
# #         change_percent = (change / prev_close) * 100

# #         prices = [float(p) for p in hist["Close"].tail(30).tolist()]

# #         ma50 = float(hist["Close"].rolling(50).mean().iloc[-1])

# #         delta = hist["Close"].diff()
# #         gain = (delta.where(delta > 0, 0)).rolling(14).mean()
# #         loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
# #         rsi = float(100 - (100 / (1 + (gain / loss))).iloc[-1])

# #         # Technical Signal
# #         if current_price > ma50:
# #             trend = "Uptrend"
# #             signal = "BUY" if rsi < 70 else "NEUTRAL"
# #         else:
# #             trend = "Downtrend"
# #             signal = "SELL" if rsi > 30 else "NEUTRAL"

# #         return {
# #             "current_price": current_price,
# #             "change": change,
# #             "change_percent": change_percent,
# #             "rsi": rsi,
# #             "ma50": ma50,
# #             "trend": trend,
# #             "signal": signal,
# #             "prices": prices
# #         }
# #     except Exception as e:
# #         print("❌ Technical error:", e)
# #         return None


# # # ---------------------------------------------------------
# # # 3. MODELS
# # # ---------------------------------------------------------
# # class StockRequest(BaseModel):
# #     ticker: str


# # class Holding(BaseModel):
# #     ticker: str
# #     quantity: float
# #     buyPrice: float


# # class PortfolioRequest(BaseModel):
# #     holdings: List[Holding]


# # # ---------------------------------------------------------
# # # 4. API — STOCK PREDICTION
# # # ---------------------------------------------------------
# # @app.post("/api/predict")
# # async def predict_stock(request: StockRequest):
# #     ticker = request.ticker.upper().strip()

# #     tech = analyze_technicals(ticker)

# #     if not tech and "." not in ticker:
# #         # Try Indian market
# #         ticker = ticker + ".NS"
# #         tech = analyze_technicals(ticker)

# #     if not tech:
# #         raise HTTPException(
# #             status_code=404, detail=f"No stock data for {request.ticker}")

# #     # --------------------- NEWS ----------------------
# #     clean = ticker.replace(".NS", "").replace(".BO", "")
# #     region = "IN" if ticker.endswith((".NS", ".BO")) else "US"

# #     try:
# #         g = GNews(language='en', country=region, period='7d', max_results=5)
# #         news_items = g.get_news(clean)
# #     except:
# #         news_items = []

# #     news_list = []
# #     total_sent_score = 0

# #     for n in news_items:
# #         title = n.get("title", "")
# #         scr, lbl = get_sentiment(title)
# #         total_sent_score += scr

# #         news_list.append({
# #             "title": title,
# #             "source": n.get("publisher", {}).get("title", "Unknown"),
# #             "sentiment": lbl,
# #             "url": n.get("url", "#")
# #         })

# #     avg_score = total_sent_score / len(news_items) if news_items else 0

# #     if avg_score > 0.15:
# #         sentiment_label = "Positive"
# #     elif avg_score < -0.15:
# #         sentiment_label = "Negative"
# #     else:
# #         sentiment_label = "Neutral"

# #     # ------------------ AI VERDICT -------------------
# #     tech_sig = tech["signal"]

# #     if tech_sig == "BUY" and sentiment_label == "Positive":
# #         ai_signal = "BUY"
# #         tp = tech["current_price"] * 1.05
# #         conf = 0.85

# #     elif tech_sig == "SELL" and sentiment_label == "Negative":
# #         ai_signal = "SELL"
# #         tp = tech["current_price"] * 0.95
# #         conf = 0.85

# #     else:
# #         ai_signal = "HOLD"
# #         tp = tech["current_price"] * 1.01
# #         conf = 0.60

# #     return {
# #         "ticker": ticker,
# #         "name": ticker,
# #         "currentPrice": round(tech["current_price"], 2),
# #         "change": round(tech["change"], 2),
# #         "changePercent": round(tech["change_percent"], 2),
# #         "prices": tech["prices"],

# #         "prediction": {
# #             "targetPrice": round(tp, 2),
# #             "confidence": conf,
# #             "timeframe": "7 Days",
# #             "signal": ai_signal
# #         },

# #         "sentiment": {
# #             "score": round(avg_score, 2),
# #             "label": sentiment_label
# #         },

# #         "news": news_list
# #     }


# # # ---------------------------------------------------------
# # # 5. API — PORTFOLIO ANALYSIS
# # # ---------------------------------------------------------
# # @app.post("/api/portfolio/analyze")
# # async def analyze_portfolio(request: PortfolioRequest):
# #     if not request.holdings:
# #         raise HTTPException(status_code=400, detail="No holdings given")

# #     results = []
# #     total_invested = 0
# #     total_value = 0

# #     for h in request.holdings:
# #         try:
# #             stock = await predict_stock(StockRequest(ticker=h.ticker))
# #         except:
# #             continue  # Skip invalid stocks

# #         invested = h.buyPrice * h.quantity
# #         value = stock["currentPrice"] * h.quantity
# #         pnl = value - invested
# #         pnl_percent = (pnl / invested * 100) if invested else 0

# #         total_invested += invested
# #         total_value += value

# #         results.append({
# #             "ticker": stock["ticker"],
# #             "quantity": h.quantity,
# #             "buyPrice": h.buyPrice,
# #             "currentPrice": stock["currentPrice"],
# #             "invested": round(invested, 2),
# #             "value": round(value, 2),
# #             "pnl": round(pnl, 2),
# #             "pnlPercent": round(pnl_percent, 2),
# #             "prediction": stock["prediction"],
# #             "sentiment": stock["sentiment"],
# #         })

# #     if not results:
# #         raise HTTPException(status_code=404, detail="No valid holdings")

# #     total_pnl = total_value - total_invested

# #     overview = {
# #         "totalInvested": round(total_invested, 2),
# #         "currentValue": round(total_value, 2),
# #         "totalPnL": round(total_pnl, 2),
# #         "totalPnLPercent": round(total_pnl / total_invested * 100, 2)
# #         if total_invested else 0
# #     }

# #     for r in results:
# #         r["weight"] = round((r["value"] / total_value * 100)
# #                             if total_value else 0, 2)

# #     return {"overview": overview, "stocks": results}


# # # ---------------------------------------------------------
# # # 6. RUN SERVER
# # # ---------------------------------------------------------
# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)


# #mast code

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import yfinance as yf
from gnews import GNews
from yahooquery import search
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Market Pulse Backend - AWS ECS Deployment
import os
import requests
from database import init_db, User, Holding as DBHolding

# Initialize the database and create tables
try:
    init_db()
except Exception as e:
    print("DB Init Error:", e)

# Internal Lambda API URLs (Set these in ECS environment variables later)
LAMBDA_API_URL = os.environ.get("LAMBDA_API_URL")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to Market Pulse API v1.0.1", "status": "Online"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize lightweight local VADER and setup score map
vader_analyzer = SentimentIntensityAnalyzer()
score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
HF_API_URL = "https://api-inference.huggingface.co/models/yiyanghkust/finbert-tone"

def get_vader_sentiment(text: str):
    try:
        vs = vader_analyzer.polarity_scores(text)
        compound = vs['compound']
        if compound >= 0.05:
            return 1, "Positive"
        elif compound <= -0.05:
            return -1, "Negative"
        else:
            return 0, "Neutral"
    except Exception as e:
        print("Vader Sentiment Error:", e)
        return 0, "Neutral"

#
def resolve_ticker(query: str):
    """Finds the best ticker symbol for a given name/query"""
    try:
        results = search(query)
        if 'quotes' in results and len(results['quotes']) > 0:
            # Prefer EQUITY types
            quotes = [q for q in results['quotes'] if q.get('quoteType') == 'EQUITY']
            if not quotes:
                quotes = results['quotes']
            
            best_match = quotes[0]['symbol']
            print(f"Resolved '{query}' to '{best_match}'")
            return best_match
    except Exception as e:
        print(f"Search error: {e}")
    return query


def get_sentiment(text: str):
    # Try Hugging Face Inference API first (free and fast)
    try:
        response = requests.post(HF_API_URL, json={"inputs": text[:512]}, timeout=3)
        if response.status_code == 200:
            res_data = response.json()
            if isinstance(res_data, list) and len(res_data) > 0:
                predictions = res_data[0]
                best_pred = max(predictions, key=lambda x: x["score"])
                label = best_pred["label"].capitalize()
                return score_map.get(label, 0), label
    except Exception as e:
        print("HF API Sentiment Error, falling back to VADER:", e)
    
    return get_vader_sentiment(text)


def analyze_technicals(ticker: str):
    try:
        hist_data = None
        # 1. Try fetching from internal Lambda API first
        if LAMBDA_API_URL:
            try:
                res = requests.get(f"{LAMBDA_API_URL}/price/{ticker}", timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    if "history" in data:
                        import pandas as pd
                        hist_data = pd.DataFrame({"Close": data["history"]})
            except Exception as api_err:
                print(f"Lambda API error for price: {api_err}")

        # 2. Fallback to direct Yahoo Finance call if Lambda failed or is missing
        if hist_data is None:
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period="6mo")

        if len(hist_data) < 50:
            return None

        current_price = float(hist_data["Close"].iloc[-1])
        prev_close = float(hist_data["Close"].iloc[-2])
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100

        prices = [float(p) for p in hist_data["Close"].tail(30).tolist()]

        ma50 = float(hist_data["Close"].rolling(50).mean().iloc[-1])

        delta = hist_data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = float(100 - (100 / (1 + (gain / loss))).iloc[-1])

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
        print("Technical error:", e)
        return None

#
class StockRequest(BaseModel):
    ticker: str


class Holding(BaseModel):
    ticker: str
    quantity: float
    buyPrice: float


class PortfolioRequest(BaseModel):
    holdings: List[Holding]


#
@app.post("/api/predict")
async def predict_stock(request: StockRequest):
    query = request.ticker.strip()
    
    # Try resolving if it doesn't look like a ticker
    ticker = resolve_ticker(query)

    tech = analyze_technicals(ticker)

    if not tech and "." not in ticker:
        # Try Indian market fallback
        ticker_ns = ticker + ".NS"
        tech = analyze_technicals(ticker_ns)
        if tech:
            ticker = ticker_ns

    if not tech:
        # Try one more time with original query if resolution failed or led to no data
        tech = analyze_technicals(query)
        if tech:
            ticker = query

    if not tech:
        raise HTTPException(
            status_code=404, detail=f"No stock data for {request.ticker}")

    # --------------------- NEWS ----------------------
    clean = ticker.replace(".NS", "").replace(".BO", "")
    region = "IN" if ticker.endswith((".NS", ".BO")) else "US"

    news_items = []
    
    # 1. Try Lambda API first
    if LAMBDA_API_URL:
        try:
            res = requests.get(f"{LAMBDA_API_URL}/news/{clean}", timeout=5)
            if res.status_code == 200:
                data = res.json()
                if "news" in data:
                    for n in data["news"]:
                        news_items.append({"published date": n[0], "title": n[1]})
        except Exception as api_err:
            print(f"Lambda API error for news: {api_err}")

    # 2. Fallback to direct GNews
    if not news_items:
        try:
            g = GNews(language='en', country=region, period='7d', max_results=5)
            news_items = g.get_news(clean)
        except:
            news_items = []

    news_list = []
    total_sent_score = 0

    for n in news_items:
        title = n.get("title", "")
        scr, lbl = get_sentiment(title)
        total_sent_score += scr

        news_list.append({
            "title": title,
            "source": n.get("publisher", {}).get("title", "Unknown") if "publisher" in n else "Unknown",
            "sentiment": lbl,
            "url": n.get("url", "#")
        })

    avg_score = total_sent_score / len(news_items) if news_items else 0

    if avg_score > 0.15:
        sentiment_label = "Positive"
    elif avg_score < -0.15:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # ------------------ AI VERDICT -------------------
    tech_sig = tech["signal"]

    if tech_sig == "BUY" and sentiment_label == "Positive":
        ai_signal = "BUY"
        tp = tech["current_price"] * 1.05
        conf = 0.85

    elif tech_sig == "SELL" and sentiment_label == "Negative":
        ai_signal = "SELL"
        tp = tech["current_price"] * 0.95
        conf = 0.85

    else:
        ai_signal = "HOLD"
        tp = tech["current_price"] * 1.01
        conf = 0.60

    return {
        "ticker": ticker,
        "name": ticker,
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


#
@app.post("/api/portfolio/analyze")
async def analyze_portfolio(request: PortfolioRequest):
    if not request.holdings:
        raise HTTPException(status_code=400, detail="No holdings given")

    results = []
    total_invested = 0
    total_value = 0

    for h in request.holdings:
        try:
            stock = await predict_stock(StockRequest(ticker=h.ticker))
        except:
            continue  # Skip invalid stocks

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
        raise HTTPException(status_code=404, detail="No valid holdings")

    total_pnl = total_value - total_invested

    overview = {
        "totalInvested": round(total_invested, 2),
        "currentValue": round(total_value, 2),
        "totalPnL": round(total_pnl, 2),
        "totalPnLPercent": round(total_pnl / total_invested * 100, 2)
        if total_invested else 0
    }

    for r in results:
        r["weight"] = round((r["value"] / total_value * 100)
                            if total_value else 0, 2)

    return {"overview": overview, "stocks": results}


# 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ## mast code end

