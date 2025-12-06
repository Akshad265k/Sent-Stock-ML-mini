// // export interface StockPrice {
// //   date: string;
// //   open: number;
// //   high: number;
// //   low: number;
// //   close: number;
// //   volume: number;
// // }

// // export interface NewsArticle {
// //   id: string;
// //   headline: string;
// //   source: string;
// //   publishedAt: string;
// //   sentiment: number; // -1 to 1
// //   url?: string;
// // }

// // export interface Prediction {
// //   direction: "UP" | "DOWN" | "NEUTRAL";
// //   confidence: number; // 0 to 100
// //   targetPrice?: number;
// //   change?: number; // percentage
// // }

// // export interface SentimentData {
// //   overall: number; // -1 to 1
// //   positive: number;
// //   negative: number;
// //   neutral: number;
// // }

// // export interface StockData {
// //   ticker: string;
// //   name: string;
// //   currentPrice: number;
// //   change: number;
// //   changePercent: number;
// //   prices: StockPrice[];
// //   prediction: Prediction;
// //   sentiment: SentimentData;
// //   news: NewsArticle[];
// // }


// export interface StockPrice {
//   date: string;
//   open: number;
//   high: number;
//   low: number;
//   close: number;
//   volume: number;
// }

// export interface NewsArticle {
//   title: string;
//   source: string;
//   sentiment: string;
//   url: string;
//   publishedAt?: string;
// }

// export interface StockPrediction {
//   targetPrice: number;
//   confidence: number;
//   timeframe: string;
//   signal: string; // This was missing and causing the PredictionCard error
// }

// export interface StockSentiment {
//   score: number;
//   label: string;
// }

// export interface StockData {
//   ticker: string;
//   name: string;
//   currentPrice: number;
//   change: number;
//   changePercent: number;
//   prices: number[]; // Fixed: It's an array of numbers, not objects
//   prediction: StockPrediction;
//   sentiment: StockSentiment;
//   news: NewsItem[];
// }

// // Alias NewsItem to NewsArticle if needed for compatibility
// export type NewsItem = NewsArticle;


// //mast
// export interface StockNewsItem {
//   title: string;
//   source: string;
//   sentiment: string;
//   url: string;
// }

// export interface StockSentiment {
//   score: number;   // e.g. 0.25
//   label: string;   // "Positive", "Neutral", "Negative"
// }

// export interface StockPrediction {
//   targetPrice: number;   // e.g. 945.10
//   confidence: number;    // 0.85
//   timeframe: string;     // "7 Days"
//   signal: string;        // "BUY" / "SELL" / "NEUTRAL"
// }

// export interface StockData {
//   ticker: string;        // "TATAMOTORS.NS"
//   name: string;          // "TATAMOTORS.NS"
  
//   currentPrice: number;  // 914.20
//   change: number;        // e.g. -2.45
//   changePercent: number; // e.g. -0.21

//   prices: number[];      // [101, 102, 99, ...]  (30 items)

//   prediction: StockPrediction; 
//   sentiment: StockSentiment;
//   news: StockNewsItem[];
// }

// export interface PortfolioHolding {
//   ticker: string;
//   quantity: number;
//   buyPrice: number;
// }

// export interface PortfolioStock {
//   ticker: string;
//   quantity: number;
//   buyPrice: number;
//   currentPrice: number;
//   invested: number;
//   value: number;
//   pnl: number;
//   pnlPercent: number;
//   weight: number;
//   prediction: StockPrediction;
//   sentiment: StockSentiment;
// }

// export interface PortfolioOverview {
//   totalInvested: number;
//   currentValue: number;
//   totalPnL: number;
//   totalPnLPercent: number;
// }

// export interface PortfolioResponse {
//   overview: PortfolioOverview;
//   stocks: PortfolioStock[];
// }

// export interface PortfolioHolding {
//   ticker: string;
//   quantity: number;
//   buyPrice: number;
// }

// export interface PortfolioResponse {
//   overview: {
//     totalInvested: number;
//     currentValue: number;
//     totalPnL: number;
//     totalPnLPercent: number;
//   };
//   stocks: any[];
// }

// -----------------------------
// News + Sentiment Models
// -----------------------------
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


// -----------------------------
// Prediction Model
// -----------------------------
export interface StockPrediction {
  targetPrice: number;
  confidence: number;    
  timeframe: string;     
  signal: string;        
}


// -----------------------------
// Single Stock Data Model
// -----------------------------
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


// -----------------------------
// Portfolio Models
// -----------------------------
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
