import json
import os
import redis
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

# Initialize Redis connection outside the handler
redis_url = os.environ.get('CACHE_URL', 'localhost')
try:
    cache = redis.Redis(host=redis_url, port=6379, db=0, decode_responses=True, socket_connect_timeout=2)
except Exception:
    cache = None

GNEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

def parse_rss(xml_bytes):
    """Parse Google News RSS XML using only stdlib."""
    root = ET.fromstring(xml_bytes)
    items = []
    channel = root.find('channel')
    if channel is None:
        return items
    for item in channel.findall('item')[:10]:
        title = item.findtext('title', default='')
        pub_date = item.findtext('pubDate', default='')
        items.append([pub_date, title])
    return items

def lambda_handler(event, context):
    ticker = event.get('pathParameters', {}).get('ticker')
    if not ticker:
        return {'statusCode': 400, 'body': json.dumps({'error': 'Ticker is required'})}

    # 1. Check Redis Cache First
    cache_key = f"news:{ticker}"
    try:
        if cache:
            cached_news = cache.get(cache_key)
            if cached_news:
                return {
                    'statusCode': 200,
                    'body': json.dumps({'ticker': ticker, 'news': json.loads(cached_news), 'source': 'cache'})
                }
    except Exception as e:
        print(f"Redis cache read error: {e}")

    # 2. Fetch from Google News RSS using stdlib only (no 3rd-party deps)
    try:
        query = urllib.parse.quote(ticker)
        rss_url = GNEWS_RSS.format(query=query)

        req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_bytes = response.read()

        structured_data = parse_rss(xml_bytes)

        # 3. Store in Redis with 5-minute TTL
        try:
            if cache:
                cache.setex(cache_key, 300, json.dumps(structured_data))
        except Exception as e:
            print(f"Redis cache write error: {e}")

        return {
            'statusCode': 200,
            'body': json.dumps({'ticker': ticker, 'news': structured_data, 'source': 'google-rss'})
        }
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
