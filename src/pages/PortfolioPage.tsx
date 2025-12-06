// "use client";

// import React, { useEffect, useState } from "react";
// import { Header } from "@/components/Header";
// import {
//   PortfolioHolding,
//   PortfolioResponse,
// } from "@/types/stock";
// import { Plus, Trash2, Loader2, TrendingUp, Wallet, ArrowUpRight, ArrowDownRight } from "lucide-react";
// import { toast } from "sonner";

// const DEFAULT_HOLDINGS: PortfolioHolding[] = [
//   { ticker: "RELIANCE.NS", quantity: 5, buyPrice: 1500 },
//   { ticker: "TCS.NS", quantity: 3, buyPrice: 3700 },
// ];

// const PortfolioPage = () => {
//   const [holdings, setHoldings] = useState<PortfolioHolding[]>([]);
//   const [portfolio, setPortfolio] = useState<PortfolioResponse | null>(null);
//   const [loading, setLoading] = useState(false);

//   // Load from localStorage on mount
//   useEffect(() => {
//     const saved = typeof window !== "undefined"
//       ? window.localStorage.getItem("ai_portfolio_holdings")
//       : null;

//     if (saved) {
//       setHoldings(JSON.parse(saved));
//     } else {
//       setHoldings(DEFAULT_HOLDINGS);
//     }
//   }, []);

//   // Persist to localStorage
//   useEffect(() => {
//     if (typeof window !== "undefined") {
//       window.localStorage.setItem("ai_portfolio_holdings", JSON.stringify(holdings));
//     }
//   }, [holdings]);

//   const updateHolding = (index: number, field: keyof PortfolioHolding, value: string) => {
//     setHoldings(prev => {
//       const copy = [...prev];
//       const parsedValue =
//         field === "quantity" || field === "buyPrice"
//           ? Number(value) || 0
//           : value.toUpperCase();

//       copy[index] = { ...copy[index], [field]: parsedValue } as PortfolioHolding;
//       return copy;
//     });
//   };

//   const addRow = () => {
//     setHoldings(prev => [...prev, { ticker: "", quantity: 0, buyPrice: 0 }]);
//   };

//   const removeRow = (index: number) => {
//     setHoldings(prev => prev.filter((_, i) => i !== index));
//   };

//   const handleAnalyze = async () => {
//     const validHoldings = holdings.filter(h => h.ticker && h.quantity > 0 && h.buyPrice > 0);
//     if (validHoldings.length === 0) {
//       toast.error("Add at least one valid holding (ticker, quantity, buy price).");
//       return;
//     }

//     setLoading(true);
//     try {
//       const res = await fetch("http://localhost:8000/api/portfolio/analyze", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ holdings: validHoldings }),
//       });

//       if (!res.ok) {
//         const error = await res.json().catch(() => ({}));
//         throw new Error(error.detail || "Failed to analyze portfolio");
//       }

//       const data: PortfolioResponse = await res.json();
//       setPortfolio(data);
//       toast.success("Portfolio analyzed with AI 📈");
//     } catch (err: any) {
//       console.error(err);
//       toast.error(err.message || "Error analyzing portfolio");
//     } finally {
//       setLoading(false);
//     }
//   };

//   const formatCurrency = (value: number) => `₹${value.toFixed(2)}`;

//   return (
//     <div className="min-h-screen bg-background text-foreground">
//       <Header />

//       <main className="container mx-auto px-4 py-8 space-y-8">
//         {/* Title */}
//         <div className="flex items-center justify-between gap-4">
//           <div>
//             <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
//               <Wallet className="w-7 h-7 text-emerald-400" />
//               AI Portfolio Analyzer
//             </h1>
//             <p className="text-sm text-muted-foreground">
//               Track your holdings and get AI-powered insights across your entire portfolio.
//             </p>
//           </div>

//           <button
//             onClick={handleAnalyze}
//             disabled={loading}
//             className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 hover:bg-emerald-400
//                        disabled:opacity-60 px-4 py-2 text-sm font-semibold text-black shadow-lg
//                        transition-colors"
//           >
//             {loading ? (
//               <>
//                 <Loader2 className="w-4 h-4 animate-spin" />
//                 Analyzing…
//               </>
//             ) : (
//               <>
//                 <TrendingUp className="w-4 h-4" />
//                 Analyze Portfolio
//               </>
//             )}
//           </button>
//         </div>

//         {/* Holdings Editor */}
//         <section className="rounded-2xl border border-gray-800 bg-gray-900/60 backdrop-blur-xl shadow-xl p-6 space-y-4">
//           <div className="flex items-center justify-between">
//             <h2 className="text-lg font-semibold text-gray-100">Your Holdings</h2>
//             <button
//               onClick={addRow}
//               className="inline-flex items-center gap-1 text-xs font-medium rounded-md px-3 py-1.5
//                          border border-gray-700 hover:border-emerald-500 hover:text-emerald-300
//                          text-gray-300 transition-colors"
//             >
//               <Plus className="w-3 h-3" />
//               Add Stock
//             </button>
//           </div>

//           <div className="overflow-x-auto">
//             <table className="w-full text-sm">
//               <thead className="text-xs uppercase text-gray-500 border-b border-gray-800">
//                 <tr>
//                   <th className="py-2 text-left">Ticker</th>
//                   <th className="py-2 text-left">Quantity</th>
//                   <th className="py-2 text-left">Buy Price</th>
//                   <th className="py-2 text-right">Actions</th>
//                 </tr>
//               </thead>
//               <tbody>
//                 {holdings.map((h, index) => (
//                   <tr key={index} className="border-b border-gray-800/60">
//                     <td className="py-2 pr-3">
//                       <input
//                         value={h.ticker}
//                         onChange={e => updateHolding(index, "ticker", e.target.value)}
//                         className="w-full bg-gray-900/70 border border-gray-700 rounded-md px-2 py-1
//                                    text-gray-100 text-xs focus:outline-none focus:border-emerald-500"
//                         placeholder="RELIANCE.NS"
//                       />
//                     </td>
//                     <td className="py-2 pr-3">
//                       <input
//                         type="number"
//                         value={h.quantity || ""}
//                         onChange={e => updateHolding(index, "quantity", e.target.value)}
//                         className="w-full bg-gray-900/70 border border-gray-700 rounded-md px-2 py-1
//                                    text-gray-100 text-xs focus:outline-none focus:border-emerald-500"
//                         placeholder="5"
//                       />
//                     </td>
//                     <td className="py-2 pr-3">
//                       <input
//                         type="number"
//                         value={h.buyPrice || ""}
//                         onChange={e => updateHolding(index, "buyPrice", e.target.value)}
//                         className="w-full bg-gray-900/70 border border-gray-700 rounded-md px-2 py-1
//                                    text-gray-100 text-xs focus:outline-none focus:border-emerald-500"
//                         placeholder="1500"
//                       />
//                     </td>
//                     <td className="py-2 text-right">
//                       <button
//                         onClick={() => removeRow(index)}
//                         className="inline-flex items-center justify-center rounded-md p-1.5
//                                    text-gray-500 hover:text-rose-400 hover:bg-gray-800/80 transition-colors"
//                       >
//                         <Trash2 className="w-4 h-4" />
//                       </button>
//                     </td>
//                   </tr>
//                 ))}

//                 {holdings.length === 0 && (
//                   <tr>
//                     <td className="py-4 text-sm text-gray-500" colSpan={4}>
//                       No holdings yet. Add a stock to get started.
//                     </td>
//                   </tr>
//                 )}
//               </tbody>
//             </table>
//           </div>
//         </section>

//         {/* Portfolio Analysis */}
//         {portfolio && (
//           <section className="space-y-6">
//             {/* Overview */}
//             <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
//               <div className="rounded-2xl border border-gray-800 bg-gray-900/70 p-4">
//                 <p className="text-xs text-gray-500">Total Invested</p>
//                 <p className="text-2xl font-semibold text-gray-100">
//                   {formatCurrency(portfolio.overview.totalInvested)}
//                 </p>
//               </div>
//               <div className="rounded-2xl border border-gray-800 bg-gray-900/70 p-4">
//                 <p className="text-xs text-gray-500">Current Value</p>
//                 <p className="text-2xl font-semibold text-gray-100">
//                   {formatCurrency(portfolio.overview.currentValue)}
//                 </p>
//               </div>
//               <div className="rounded-2xl border border-gray-800 bg-gray-900/70 p-4">
//                 <p className="text-xs text-gray-500">Total P&L</p>
//                 <p className={`text-2xl font-semibold ${
//                   portfolio.overview.totalPnL >= 0 ? "text-emerald-400" : "text-rose-400"
//                 }`}>
//                   {portfolio.overview.totalPnL >= 0 ? "+" : ""}
//                   {formatCurrency(portfolio.overview.totalPnL)}{" "}
//                   <span className="text-sm text-gray-400">
//                     ({portfolio.overview.totalPnLPercent >= 0 ? "+" : ""}
//                     {portfolio.overview.totalPnLPercent.toFixed(2)}%)
//                   </span>
//                 </p>
//               </div>
//             </div>

//             {/* Per-stock Cards */}
//             <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
//               {portfolio.stocks.map(stock => {
//                 const positive = stock.pnl >= 0;
//                 const SignalIcon = stock.prediction.signal === "BUY"
//                   ? ArrowUpRight
//                   : stock.prediction.signal === "SELL"
//                   ? ArrowDownRight
//                   : TrendingUp;

//                 return (
//                   <div
//                     key={stock.ticker}
//                     className="rounded-2xl border border-gray-800 bg-gray-900/70 p-5 hover:border-emerald-500/60 transition-colors"
//                   >
//                     <div className="flex items-start justify-between mb-3">
//                       <div>
//                         <h3 className="text-lg font-semibold text-gray-100">{stock.ticker}</h3>
//                         <p className="text-xs text-gray-500">
//                           Position: {stock.quantity} @ ₹{stock.buyPrice.toFixed(2)}
//                         </p>
//                       </div>
//                       <div className="text-right">
//                         <p className="text-xs text-gray-500 mb-1">AI Signal</p>
//                         <span className="inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-semibold
//                           bg-gray-800 border border-gray-700 text-gray-200"
//                         >
//                           <SignalIcon className="w-3 h-3" />
//                           {stock.prediction.signal}
//                         </span>
//                       </div>
//                     </div>

//                     <div className="grid grid-cols-2 gap-3 text-sm mb-3">
//                       <div>
//                         <p className="text-gray-500 text-xs">Current Price</p>
//                         <p className="text-gray-100 font-medium">
//                           {formatCurrency(stock.currentPrice)}
//                         </p>
//                       </div>
//                       <div>
//                         <p className="text-gray-500 text-xs">Target Price</p>
//                         <p className="text-gray-100 font-medium">
//                           {formatCurrency(stock.prediction.targetPrice)}
//                         </p>
//                       </div>
//                       <div>
//                         <p className="text-gray-500 text-xs">Position Value</p>
//                         <p className="text-gray-100 font-medium">
//                           {formatCurrency(stock.value)}
//                         </p>
//                       </div>
//                       <div>
//                         <p className="text-gray-500 text-xs">Weight</p>
//                         <p className="text-gray-100 font-medium">
//                           {stock.weight.toFixed(1)}%
//                         </p>
//                       </div>
//                     </div>

//                     <div className="flex items-center justify-between text-sm">
//                       <div>
//                         <p className="text-gray-500 text-xs">P&L</p>
//                         <p className={`${positive ? "text-emerald-400" : "text-rose-400"} font-semibold`}>
//                           {positive ? "+" : ""}
//                           {formatCurrency(stock.pnl)}{" "}
//                           <span className="text-xs text-gray-400">
//                             ({positive ? "+" : ""}{stock.pnlPercent.toFixed(2)}%)
//                           </span>
//                         </p>
//                       </div>
//                       <div className="text-right">
//                         <p className="text-gray-500 text-xs">News Sentiment</p>
//                         <p className="text-xs text-gray-300">
//                           {stock.sentiment.label} ({stock.sentiment.score.toFixed(2)})
//                         </p>
//                       </div>
//                     </div>
//                   </div>
//                 );
//               })}
//             </div>
//           </section>
//         )}
//       </main>
//     </div>
//   );
// };

// export default PortfolioPage;


"use client";

import { useEffect, useState } from "react";
import { PortfolioHolding, PortfolioResponse } from "@/types/stock";
import { TrendingUp, Wallet, Percent, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { Header } from "@/components/Header";

export default function PortfolioPage() {
  const [portfolio, setPortfolio] = useState<PortfolioHolding[]>([]);
  const [stats, setStats] = useState<PortfolioResponse | null>(null);

  // Load holdings from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("ai_portfolio_holdings");
    if (saved) setPortfolio(JSON.parse(saved));
  }, []);

  // Fetch backend analysis
  useEffect(() => {
    if (portfolio.length === 0) return;

    const fetchStats = async () => {
      const res = await fetch("http://localhost:8000/api/portfolio/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ holdings: portfolio }),
      });

      if (!res.ok) return;
      const data = await res.json();
      setStats(data);
    };

    fetchStats();
  }, [portfolio]);

  if (!portfolio.length)
    return (
      <div className="p-10 text-center text-gray-400">
        Your portfolio is empty. Analyze a stock and add it to your portfolio.
      </div>
    );

  if (!stats)
    return (
      <div className="p-10 text-center text-gray-400">
        Loading portfolio…
      </div>
    );

  const { overview, stocks } = stats;

  const gainColor = overview.totalPnL >= 0 ? "text-emerald-400" : "text-rose-400";

  return (
    <div className="p-10 space-y-10 text-gray-100">
        <Header />
      {/* PAGE HEADER */}
      <div>
        <h1 className="text-3xl font-bold mb-1">Portfolio</h1>
        <p className="text-gray-400">Track your stock holdings and performance</p>
      </div>

      {/* SUMMARY CARDS */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

        {/* Total Value */}
        <div className="bg-gray-900/70 border border-gray-800 rounded-xl p-6 shadow-sm">
          <p className="text-gray-400 mb-1">Total Value</p>
          <p className="text-4xl font-bold">₹{overview.currentValue.toLocaleString()}</p>
        </div>

        {/* Total Gain/Loss */}
        <div className="bg-gray-900/70 border border-gray-800 rounded-xl p-6 shadow-sm">
          <p className="text-gray-400 mb-1">Total Gain/Loss</p>
          <div className={`text-2xl font-bold flex items-center gap-2 ${gainColor}`}>
            {overview.totalPnL >= 0 ? <ArrowUpRight /> : <ArrowDownRight />}
            {overview.totalPnL >= 0 ? "+" : ""}
            ₹{overview.totalPnL.toLocaleString()}
          </div>
        </div>

        {/* Return (%) */}
        <div className="bg-gray-900/70 border border-gray-800 rounded-xl p-6 shadow-sm">
          <p className="text-gray-400 mb-1">Return</p>
          <div className={`text-2xl font-bold flex items-center gap-2 ${gainColor}`}>
            <Percent className="h-5 w-5" />
            {overview.totalPnLPercent >= 0 ? "+" : ""}
            {overview.totalPnLPercent}%
          </div>
        </div>
      </div>

      {/* HOLDINGS TABLE */}
      <div className="bg-gray-900/70 border border-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-xl font-semibold mb-6">Holdings</h2>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-800">
                <th className="py-2">Stock</th>
                <th className="py-2">Shares</th>
                <th className="py-2">Avg Price</th>
                <th className="py-2">Current Price</th>
                <th className="py-2">Market Value</th>
                <th className="py-2">Gain/Loss</th>
              </tr>
            </thead>

            <tbody>
              {stocks.map((s) => {
                const color = s.pnl >= 0 ? "text-emerald-400" : "text-rose-400";

                return (
                  <tr key={s.ticker} className="border-b border-gray-800">
                    <td className="py-3">
                      <div className="font-semibold">{s.ticker}</div>
                      <div className="text-xs text-gray-500">{s.sentiment.label}</div>
                    </td>

                    <td className="py-3">{s.quantity}</td>

                    <td className="py-3">
                      ₹{s.buyPrice.toLocaleString()}
                    </td>

                    <td className="py-3">
                      ₹{s.currentPrice.toLocaleString()}
                    </td>

                    <td className="py-3 font-semibold">
                      ₹{s.value.toLocaleString()}
                    </td>

                    <td className={`py-3 font-semibold ${color}`}>
                      {s.pnl >= 0 ? "+" : ""}
                      ₹{s.pnl.toLocaleString()}
                      <div className="text-xs">
                        ({s.pnlPercent >= 0 ? "+" : ""}
                        {s.pnlPercent}%)
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>

          </table>
        </div>
      </div>
    </div>
  );
}
