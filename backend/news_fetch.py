from gnews import GNews

def get_stock_news(stock_name):
    # Setup: English language, India region, last 7 days of news
    google_news = GNews(language='en', country='IN', period='7d', max_results=10)
    
    print(f"--- Fetching news for: {stock_name} ---")
    
    # Fetch news
    news_results = google_news.get_news(stock_name)
    
    if not news_results:
        print("No news found.")
        return []

    structured_data = []
    
    for item in news_results:
        print(f"\n📰 {item['publisher']['title']}")
        print(f"   {item['title']}")
        print(f"   🔗 {item['url']}")
        print(f"   📅 {item['published date']}")
        
        # Keep only what we need for the ML model later
        structured_data.append([item['published date'], item['title']])
        
    return structured_data

# Test it immediately
stock = input("Enter Stock Name (e.g., Tata Motors, Reliance, Infosys): ")
data = get_stock_news(stock)