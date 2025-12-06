import { StockData } from "@/types/stock";

export const mockStockData: Record<string, StockData> = {
  AAPL: {
    ticker: "AAPL",
    name: "Apple Inc.",
    currentPrice: 178.52,
    change: 2.34,
    changePercent: 1.33,
    prices: generateMockPrices(178.52, 60),
    prediction: {
      direction: "UP",
      confidence: 72,
      targetPrice: 182.50,
      change: 2.23,
    },
    sentiment: {
      overall: 0.65,
      positive: 68,
      negative: 18,
      neutral: 14,
    },
    news: [
      {
        id: "1",
        headline: "Apple announces new AI features for iOS 18",
        source: "TechCrunch",
        publishedAt: "2 hours ago",
        sentiment: 0.8,
      },
      {
        id: "2",
        headline: "iPhone 16 sales exceed analyst expectations in Q4",
        source: "Reuters",
        publishedAt: "4 hours ago",
        sentiment: 0.7,
      },
      {
        id: "3",
        headline: "Supply chain concerns for Apple in Asian markets",
        source: "Bloomberg",
        publishedAt: "6 hours ago",
        sentiment: -0.4,
      },
      {
        id: "4",
        headline: "Apple Vision Pro sees strong enterprise adoption",
        source: "Wall Street Journal",
        publishedAt: "8 hours ago",
        sentiment: 0.6,
      },
      {
        id: "5",
        headline: "Apple services revenue hits new record high",
        source: "CNBC",
        publishedAt: "12 hours ago",
        sentiment: 0.9,
      },
    ],
  },
  MSFT: {
    ticker: "MSFT",
    name: "Microsoft Corporation",
    currentPrice: 412.78,
    change: -1.52,
    changePercent: -0.37,
    prices: generateMockPrices(412.78, 60),
    prediction: {
      direction: "UP",
      confidence: 68,
      targetPrice: 420.00,
      change: 1.75,
    },
    sentiment: {
      overall: 0.52,
      positive: 62,
      negative: 24,
      neutral: 14,
    },
    news: [
      {
        id: "1",
        headline: "Microsoft Cloud revenue grows 28% year-over-year",
        source: "Reuters",
        publishedAt: "1 hour ago",
        sentiment: 0.85,
      },
      {
        id: "2",
        headline: "Azure AI services gain major enterprise clients",
        source: "Bloomberg",
        publishedAt: "3 hours ago",
        sentiment: 0.7,
      },
      {
        id: "3",
        headline: "EU regulators reviewing Microsoft's AI practices",
        source: "Financial Times",
        publishedAt: "5 hours ago",
        sentiment: -0.3,
      },
      {
        id: "4",
        headline: "Gaming division shows mixed results in latest quarter",
        source: "The Verge",
        publishedAt: "9 hours ago",
        sentiment: 0.1,
      },
    ],
  },
  GOOGL: {
    ticker: "GOOGL",
    name: "Alphabet Inc.",
    currentPrice: 142.65,
    change: 0.89,
    changePercent: 0.63,
    prices: generateMockPrices(142.65, 60),
    prediction: {
      direction: "DOWN",
      confidence: 58,
      targetPrice: 138.20,
      change: -3.12,
    },
    sentiment: {
      overall: -0.24,
      positive: 34,
      negative: 48,
      neutral: 18,
    },
    news: [
      {
        id: "1",
        headline: "Google faces new antitrust challenges in search market",
        source: "Wall Street Journal",
        publishedAt: "1 hour ago",
        sentiment: -0.7,
      },
      {
        id: "2",
        headline: "YouTube ad revenue misses quarterly expectations",
        source: "CNBC",
        publishedAt: "4 hours ago",
        sentiment: -0.5,
      },
      {
        id: "3",
        headline: "Google Cloud announces partnership with major retailers",
        source: "TechCrunch",
        publishedAt: "7 hours ago",
        sentiment: 0.6,
      },
      {
        id: "4",
        headline: "Concerns over Google's AI competition with OpenAI",
        source: "Bloomberg",
        publishedAt: "10 hours ago",
        sentiment: -0.4,
      },
    ],
  },
};

function generateMockPrices(currentPrice: number, days: number) {
  const prices = [];
  let price = currentPrice * 0.95; // Start from 5% lower

  for (let i = days; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    
    const volatility = 0.02;
    const change = (Math.random() - 0.48) * price * volatility;
    price += change;
    
    const dayHigh = price * (1 + Math.random() * 0.01);
    const dayLow = price * (1 - Math.random() * 0.01);
    
    prices.push({
      date: date.toISOString().split('T')[0],
      open: price,
      high: dayHigh,
      low: dayLow,
      close: price,
      volume: Math.floor(Math.random() * 10000000) + 5000000,
    });
  }
  
  return prices;
}
