// ========================================
// Sent-Stock Type Definitions
// ========================================

// --- News + Sentiment ---
export interface StockNewsItem {
  title: string;
  source: string;
  sentiment: string;
  url: string;
}

export interface StockSentiment {
  score: number;     // e.g. 0.25
  label: string;     // "Positive", "Neutral", "Negative"
}

// --- Prediction ---
export interface StockPrediction {
  targetPrice: number;
  confidence: number;    
  timeframe: string;     
  signal: string;        
}

// --- Single Stock Data ---
export interface StockData {
  ticker: string;
  name: string;
  
  currentPrice: number;
  change: number;
  changePercent: number;

  prices: number[];

  prediction: StockPrediction; 
  sentiment: StockSentiment;
  news: StockNewsItem[];
}

// --- Portfolio ---
export interface PortfolioHolding {
  ticker: string;
  quantity: number;
  buyPrice: number;
}

export interface PortfolioStock {
  ticker: string;
  quantity: number;
  buyPrice: number;

  currentPrice: number;

  invested: number;     // buyPrice * quantity
  value: number;        // currentPrice * quantity

  pnl: number;          // value - invested
  pnlPercent: number;   // pnl / invested * 100

  weight: number;       // % of portfolio

  prediction: StockPrediction;
  sentiment: StockSentiment;
}

export interface PortfolioOverview {
  totalInvested: number;
  currentValue: number;
  totalPnL: number;
  totalPnLPercent: number;
}

export interface PortfolioResponse {
  overview: PortfolioOverview;
  stocks: PortfolioStock[];
}
