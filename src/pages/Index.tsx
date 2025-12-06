// import { useState } from "react";
// import { Header } from "@/components/Header";
// import { StockSearch } from "@/components/StockSearch";
// import { StockHeader } from "@/components/StockHeader";
// import { StockChart } from "@/components/StockChart";
// import { PredictionHero } from "@/components/PredictionHero";
// import { SentimentGauge } from "@/components/SentimentGauge";
// import { NewsFeed } from "@/components/NewsFeed";
// import { mockStockData } from "@/data/mockData";
// import { StockData } from "@/types/stock";
// import { toast } from "sonner";

// const Index = () => {
//   const [selectedStock, setSelectedStock] = useState<StockData | null>(mockStockData.AAPL);

//   const handleSearch = (ticker: string) => {
//     const stockData = mockStockData[ticker];
//     if (stockData) {
//       setSelectedStock(stockData);
//       toast.success(`Loaded data for ${ticker}`);
//     } else {
//       toast.error(`No data available for ${ticker}. Try AAPL, MSFT, or GOOGL.`);
//     }
//   };

//   return (
//     <div className="min-h-screen bg-background">
//       <Header />

//       <main className="container mx-auto px-4 py-8">
//         <div className="mb-8">
//           <StockSearch onSearch={handleSearch} currentTicker={selectedStock?.ticker} />
//         </div>

//         {selectedStock ? (
//           <div className="space-y-8">
//             <StockHeader
//               ticker={selectedStock.ticker}
//               name={selectedStock.name}
//               currentPrice={selectedStock.currentPrice}
//               change={selectedStock.change}
//               changePercent={selectedStock.changePercent}
//             />

//             <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
//               <div className="lg:col-span-2 space-y-6">
//                 <StockChart
//                   prices={selectedStock.prices}
//                   currentPrice={selectedStock.currentPrice}
//                   predictionPrice={selectedStock.prediction.targetPrice}
//                   ticker={selectedStock.ticker}
//                 />
//                 <NewsFeed news={selectedStock.news} />
//               </div>

//               <div className="space-y-6">
//                 <PredictionHero
//                   prediction={selectedStock.prediction}
//                   currentPrice={selectedStock.currentPrice}
//                 />
//                 <SentimentGauge sentiment={selectedStock.sentiment} />
//               </div>
//             </div>
//           </div>
//         ) : (
//           <div className="text-center py-20">
//             <p className="text-muted-foreground text-lg">
//               Search for a stock ticker to view predictions and analysis
//             </p>
//           </div>
//         )}
//       </main>
//     </div>
//   );
// };

// export default Index;

// //mast code
// import { useState } from "react";
// import { Header } from "@/components/Header";
// import { StockSearch } from "@/components/StockSearch";
// import { StockHeader } from "@/components/StockHeader";
// import { StockChart } from "@/components/StockChart";
// // import { PredictionHero } from "@/components/PredictionHero"; // <-- REMOVE OR COMMENT OUT
// import { PredictionCard } from "@/components/PredictionCard"; // <-- IMPORT NEW CARD
// import { SentimentGauge } from "@/components/SentimentGauge";
// import { NewsFeed } from "@/components/NewsFeed";
// import { StockData } from "@/types/stock";

// import { toast } from "sonner";
// import React from "react";

// const Index = () => {
//   const [selectedStock, setSelectedStock] = useState<StockData | null>(null);
//   const [loading, setLoading] = useState(false);

//   const handleSearch = async (ticker: string) => {
//     setLoading(true);
//     try {
//       const response = await fetch('http://localhost:8000/api/predict', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ ticker }),
//       });

//       if (!response.ok) throw new Error('Failed to fetch');

//       const data = await response.json();

//       // --- DEBUGGING LOG ---
//       console.log("API DATA RECEIVED:", data);
//       // Check your browser console (F12) to see this!

//       setSelectedStock(data);
//       toast.success(`Generated AI prediction for ${data.ticker}`);

//     } catch (error) {
//       console.error(error);
//       toast.error("Failed to analyze. Check console for details.");
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="min-h-screen bg-background text-foreground">
//       <Header />

//       <main className="container mx-auto px-4 py-8">
//         <div className="mb-8 max-w-2xl mx-auto">
//           <StockSearch
//             onSearch={handleSearch}
//             currentTicker={selectedStock?.ticker}
//             isLoading={loading}
//           />
//         </div>

//         {loading ? (
//            <div className="flex flex-col items-center justify-center py-20 space-y-4">
//              <div className="animate-spin h-10 w-10 border-4 border-primary border-t-transparent rounded-full"></div>
//              <p className="text-muted-foreground animate-pulse">Analyzing Market Data & News Sentiment...</p>
//            </div>
//         ) : selectedStock ? (
//           <div className="space-y-8 animate-in fade-in duration-500">

//             {/* 1. Header Section */}
//             <StockHeader
//               ticker={selectedStock.ticker}
//               name={selectedStock.name}
//               currentPrice={selectedStock.currentPrice}
//               // ✅ FIX: Use real backend data here instead of 0
//               change={selectedStock.change || 0}
//               changePercent={selectedStock.changePercent || 0}
//             />

//             <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

//               {/* 2. Main Content (Chart & News) */}
//               <div className="lg:col-span-2 space-y-6">
//                 <StockChart
//                   prices={selectedStock.prices || []}
//                   currentPrice={selectedStock.currentPrice}
//                   predictionPrice={selectedStock.prediction.targetPrice}
//                   ticker={selectedStock.ticker}
//                 />
//                 <NewsFeed news={selectedStock.news || []} />
//               </div>

//               {/* 3. Sidebar (Verdict & Sentiment) */}
//               <div className="space-y-6">
//                 {/* ✅ NEW: Using the better Card component */}
//                 <PredictionCard
//                   currentPrice={selectedStock.currentPrice}
//                   prediction={selectedStock.prediction}
//                 />

//                 <SentimentGauge sentiment={selectedStock.sentiment} />
//               </div>
//             </div>
//           </div>
//         ) : (
//           <div className="text-center py-20 bg-muted/30 rounded-lg border border-dashed">
//             <h3 className="text-lg font-semibold mb-2">Ready to Analyze</h3>
//             <p className="text-muted-foreground">
//               Enter an Indian ticker (e.g. TATAMOTORS) to generate real-time AI predictions.
//             </p>
//           </div>
//         )}
//       </main>
//     </div>
//   );
// };

// export default Index;

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

const Index = () => {
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
    window.localStorage.setItem(
      "ai_portfolio_holdings",
      JSON.stringify(updated)
    );
  };

  const handleSearch = async (ticker: string) => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
      });

      if (!response.ok) throw new Error("Failed to fetch");

      const data = await response.json();
      setSelectedStock(data);
      toast.success(`Generated AI prediction for ${data.ticker}`);
    } catch (error) {
      toast.error("Failed to analyze. Check console.");
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

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8 max-w-2xl mx-auto">
          <StockSearch
            onSearch={handleSearch}
            currentTicker={selectedStock?.ticker}
            isLoading={loading}
          />
        </div>

        {loading ? (
          <div className="flex flex-col items-center justify-center py-20 space-y-4">
            <div className="animate-spin h-10 w-10 border-4 border-primary border-t-transparent rounded-full"></div>
            <p className="text-muted-foreground animate-pulse">
              Analyzing Market Data & News Sentiment...
            </p>
          </div>
        ) : selectedStock ? (
          <div className="space-y-8 animate-in fade-in duration-500">
            {/* Stock Header */}
            <StockHeader
              ticker={selectedStock.ticker}
              name={selectedStock.name}
              currentPrice={selectedStock.currentPrice}
              change={selectedStock.change || 0}
              changePercent={selectedStock.changePercent || 0}
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left 2 columns: Chart + News */}
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

                {/* Add to Portfolio Button */}
                <button
                  onClick={() => setPortfolioModalOpen(true)}
                  className="w-full rounded-xl bg-emerald-600 hover:bg-emerald-500 text-black 
                  font-semibold py-3 transition shadow-lg"
                >
                  ➕ Add to Portfolio
                </button>

                {/* Portfolio Widget (Mini preview) */}
                <PortfolioWidget portfolio={portfolio} onDelete={deleteStock} />
              </div>
            </div>
          </div>
        ) : (
          // <div className="text-center py-20 bg-muted/30 rounded-lg border border-dashed">
          //   <h3 className="text-lg font-semibold mb-2">Ready to Analyze</h3>
          //   <p className="text-muted-foreground">
          //     Enter an Indian ticker (e.g. TATAMOTORS) to generate real-time AI predictions.
          //   </p>
          // </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
            {/* Portfolio Chart */}
            <PortfolioChart portfolio={portfolio} />

            {/* Portfolio widget */}
            <div className="flex justify-center">
              <div className="w-full max-w-md">
                <PortfolioWidget portfolio={portfolio} onDelete={deleteStock} />
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Add-to-Portfolio Modal */}
      <AddToPortfolioModal
        isOpen={isPortfolioModalOpen}
        onClose={() => setPortfolioModalOpen(false)}
        onSave={addToPortfolio}
      />
    </div>
  );
};

export default Index;
