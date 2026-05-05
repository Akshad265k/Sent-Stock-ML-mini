import json
import os
import yfinance as yf
import redis

# Initialize Redis connection outside the handler so it can be reused across invocations
redis_url = os.environ.get('CACHE_URL', 'localhost')
cache = redis.Redis(host=redis_url, port=6379, db=0, decode_responses=True)

def lambda_handler(event, context):
    ticker = event.get('pathParameters', {}).get('ticker')
    if not ticker:
        return {'statusCode': 400, 'body': json.dumps({'error': 'Ticker is required'})}

    # 1. Check Redis Cache First
    cache_key = f"price:{ticker}"
    try:
        cached_price = cache.get(cache_key)
        if cached_price:
            return {
                'statusCode': 200,
                'body': json.dumps({'ticker': ticker, 'price': float(cached_price), 'source': 'cache'})
            }
    except Exception as e:
        print(f"Redis cache read error: {e}")

    # 2. Fetch from Yahoo Finance if not in cache
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="6mo")
        if data.empty:
            return {'statusCode': 404, 'body': json.dumps({'error': 'No data found for ticker'})}
        
        prices = [float(p) for p in data["Close"].tolist()]
        current_price = prices[-1]
        
        result_data = {
            'ticker': ticker, 
            'price': current_price, 
            'history': prices,
            'source': 'yfinance'
        }
        
        # 3. Store in Redis with 5-minute TTL (300 seconds)
        try:
            cache.setex(cache_key, 300, json.dumps(result_data))
        except Exception as e:
            print(f"Redis cache write error: {e}")
        
        return {
            'statusCode': 200,
            'body': json.dumps(result_data)
        }
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
