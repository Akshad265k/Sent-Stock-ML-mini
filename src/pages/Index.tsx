"use client";

import { useState, useEffect } from "react";
import { Header } from "@/components/Header";
import { StockSearch } from "@/components/StockSearch";
import { StockHeader } from "@/components/StockHeader";
import { StockChart } from "@/components/StockChart";
import { PredictionCard } from "@/components/PredictionCard";
import { SentimentGauge } from "@/components/SentimentGauge";
import { NewsFeed } from "@/components/NewsFeed";
import { StockData, PortfolioHolding } from "@/types/stock";
import { toast } from "sonner";
import AddToPortfolioModal from "@/components/AddToPortfolioModal";
import PortfolioWidget from "@/components/PortfolioWidget";
import PortfolioChart from "@/components/PortfolioChart";
import { motion } from "framer-motion";
import { Brain, TrendingUp, Newspaper, Zap, BarChart3 } from "lucide-react";

const Index = () => {
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";
  const [selectedStock, setSelectedStock] = useState<StockData | null>(null);
  const [loading, setLoading] = useState(false);
  const [isPortfolioModalOpen, setPortfolioModalOpen] = useState(false);
  const [portfolio, setPortfolio] = useState<PortfolioHolding[]>([]);

  const deleteStock = (ticker: string) => {
    const updated = portfolio.filter((p) => p.ticker !== ticker);
    savePortfolio(updated);
    toast.success(`${ticker} removed from portfolio`);
  };

  // Load from localStorage on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = window.localStorage.getItem("ai_portfolio_holdings");
      if (saved) setPortfolio(JSON.parse(saved));
    }
  }, []);

  // Save back to localStorage
  const savePortfolio = (updated: PortfolioHolding[]) => {
    setPortfolio(updated);
    window.localStorage.setItem("ai_portfolio_holdings", JSON.stringify(updated));
  };

  const handleSearch = async (ticker: string) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
      });

      if (!response.ok) throw new Error("Failed to fetch");

      const data = await response.json();
      setSelectedStock(data);
      toast.success(`Generated AI prediction for ${data.ticker}`);
    } catch (error) {
      toast.error("Failed to analyze. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const addToPortfolio = (quantity: number, buyPrice: number) => {
    if (!selectedStock) return;

    const newHolding: PortfolioHolding = {
      ticker: selectedStock.ticker,
      quantity,
      buyPrice,
    };

    const updated = [...portfolio, newHolding];
    savePortfolio(updated);

    toast.success(`${selectedStock.ticker} added to portfolio`);
    setPortfolioModalOpen(false);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header />

      {/* Ambient background effects */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-primary/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-emerald-500/5 rounded-full blur-[100px]" />
      </div>

      <main className="container mx-auto px-4 py-8">
        {/* Search */}
        <div className="mb-8 max-w-2xl mx-auto">
          <StockSearch
            onSearch={handleSearch}
            currentTicker={selectedStock?.ticker}
            isLoading={loading}
          />
        </div>

        {loading ? (
          /* Loading State */
          <div className="flex flex-col items-center justify-center py-20 space-y-6">
            <div className="relative">
              <div className="w-16 h-16 border-4 border-primary/20 rounded-full" />
              <div className="absolute inset-0 w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin" />
            </div>
            <div className="text-center space-y-2">
              <p className="text-foreground font-medium">Analyzing Market Data</p>
              <p className="text-muted-foreground text-sm animate-pulse">
                Processing news sentiment & technical indicators...
              </p>
            </div>
          </div>
        ) : selectedStock ? (
          /* Stock Results */
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            className="space-y-6"
          >
            <StockHeader
              ticker={selectedStock.ticker}
              name={selectedStock.name}
              currentPrice={selectedStock.currentPrice}
              change={selectedStock.change || 0}
              changePercent={selectedStock.changePercent || 0}
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left: Chart + News */}
              <div className="lg:col-span-2 space-y-6">
                <StockChart
                  prices={selectedStock.prices || []}
                  currentPrice={selectedStock.currentPrice}
                  predictionPrice={selectedStock.prediction.targetPrice}
                  ticker={selectedStock.ticker}
                />
                <NewsFeed news={selectedStock.news || []} />
              </div>

              {/* Right Sidebar */}
              <div className="space-y-6">
                <PredictionCard
                  currentPrice={selectedStock.currentPrice}
                  prediction={selectedStock.prediction}
                />

                <SentimentGauge sentiment={selectedStock.sentiment} />

                {/* Add to Portfolio */}
                <button
                  onClick={() => setPortfolioModalOpen(true)}
                  className="w-full rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-400
                    hover:from-emerald-400 hover:to-emerald-300 text-black font-semibold py-3.5
                    transition-all shadow-lg shadow-emerald-500/20 hover:shadow-emerald-500/30
                    hover:scale-[1.02] active:scale-[0.98] flex items-center justify-center gap-2"
                  id="add-portfolio-btn"
                >
                  <span className="text-lg">➕</span> Add to Portfolio
                </button>

                <PortfolioWidget portfolio={portfolio} onDelete={deleteStock} />
              </div>
            </div>
          </motion.div>
        ) : (
          /* Hero / Empty State */
          <div className="space-y-12">
            {/* Hero Section */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-center py-12"
            >
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium mb-6">
                <Zap className="w-3.5 h-3.5" />
                AI-Powered Analysis for Indian Markets
              </div>

              <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-4 leading-tight">
                Predict Stock Trends with
                <br />
                <span className="gradient-text">AI Sentiment Analysis</span>
              </h2>

              <p className="text-muted-foreground text-lg max-w-2xl mx-auto mb-8 leading-relaxed">
                Enter any Indian stock ticker to get real-time predictions powered by
                NLP-driven news sentiment analysis and technical indicators.
              </p>

              {/* Feature cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto">
                {[
                  { icon: Brain, title: "AI Predictions", desc: "ML models analyze sentiment & price patterns" },
                  { icon: Newspaper, title: "Live News Feed", desc: "Real-time news with sentiment scoring" },
                  { icon: BarChart3, title: "Technical Analysis", desc: "RSI, MA-50, and price history charts" },
                ].map((feature, i) => (
                  <motion.div
                    key={feature.title}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.2 + i * 0.1 }}
                    className="glass rounded-xl p-5 text-center hover:glow-primary transition-all duration-300"
                  >
                    <feature.icon className="w-8 h-8 text-primary mx-auto mb-3" />
                    <h3 className="font-semibold text-foreground text-sm mb-1">{feature.title}</h3>
                    <p className="text-xs text-muted-foreground">{feature.desc}</p>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* Portfolio section (if user has holdings) */}
            {portfolio.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.5 }}
                className="grid grid-cols-1 lg:grid-cols-2 gap-8"
              >
                <PortfolioChart portfolio={portfolio} />
                <div className="flex justify-center">
                  <div className="w-full max-w-md">
                    <PortfolioWidget portfolio={portfolio} onDelete={deleteStock} />
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 mt-auto">
        <div className="container mx-auto px-4 py-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-primary" />
            <span className="text-sm text-muted-foreground">
              Sent-Stock © {new Date().getFullYear()} — For educational purposes only
            </span>
          </div>
          <p className="text-xs text-muted-foreground/60">
            Not financial advice. Always consult a qualified advisor.
          </p>
        </div>
      </footer>

      {/* Modal */}
      <AddToPortfolioModal
        isOpen={isPortfolioModalOpen}
        onClose={() => setPortfolioModalOpen(false)}
        onSave={addToPortfolio}
        currentPrice={selectedStock?.currentPrice}
      />
    </div>
  );
};

export default Index;
