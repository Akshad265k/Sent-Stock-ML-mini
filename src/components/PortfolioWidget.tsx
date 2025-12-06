// import React, { useEffect, useState } from "react";
// import { PortfolioHolding, PortfolioResponse } from "@/types/stock";
// import { TrendingUp, Wallet } from "lucide-react";

// interface Props {
//   portfolio: PortfolioHolding[];
// }

// const PortfolioWidget = ({ portfolio }: Props) => {
//   const [stats, setStats] = useState<PortfolioResponse | null>(null);

//   useEffect(() => {
//     if (portfolio.length === 0) return;

//     const loadStats = async () => {
//       try {
//         const res = await fetch("http://localhost:8000/api/portfolio/analyze", {
//           method: "POST",
//           headers: { "Content-Type": "application/json" },
//           body: JSON.stringify({ holdings: portfolio }),
//         });
//         if (!res.ok) return;
//         const data = await res.json();
//         setStats(data);
//       } catch {}
//     };

//     loadStats();
//   }, [portfolio]);

//   if (portfolio.length === 0)
//     return (
//       <div className="rounded-xl bg-gray-900/80 border border-gray-800 p-6 text-center text-gray-400">
//         No stocks in your portfolio yet.
//       </div>
//     );

//   if (!stats)
//     return (
//       <div className="rounded-xl bg-gray-900/80 border border-gray-800 p-6 text-center text-gray-400">
//         Loading portfolio…
//       </div>
//     );

//   const gainColor =
//     stats.overview.totalPnL >= 0 ? "text-emerald-400" : "text-rose-400";

//   return (
//     <div className="rounded-xl bg-gray-900/80 border border-gray-800 p-6 shadow-lg space-y-6">
//       {/* Header */}
//       <div className="flex items-center gap-2">
//         <Wallet className="h-5 w-5 text-blue-400" />
//         <h2 className="text-xl font-semibold text-gray-100">
//           Your Portfolio
//         </h2>
//       </div>

//       {/* Total Value */}
//       <div>
//         <p className="text-gray-400 text-sm">Total Value</p>
//         <p className="text-3xl font-bold text-white">
//           ₹{stats.overview.currentValue.toLocaleString()}
//         </p>

//         <p className={`text-sm mt-1 flex items-center gap-1 ${gainColor}`}>
//           <TrendingUp className="h-4 w-4" />
//           {stats.overview.totalPnLPercent >= 0 ? "+" : ""}
//           {stats.overview.totalPnLPercent}% (
//           {stats.overview.totalPnL >= 0 ? "+" : ""}
//           ₹{stats.overview.totalPnL})
//         </p>
//       </div>

//       {/* Holdings List */}
//       <div className="space-y-4">
//         {stats.stocks.slice(0, 3).map((s) => {
//           const color =
//             s.pnl >= 0 ? "text-emerald-400" : "text-rose-400";

//           return (
//             <div
//               key={s.ticker}
//               className="flex items-center justify-between text-sm"
//             >
//               <div>
//                 <p className="text-gray-200 font-medium">{s.ticker}</p>
//                 <p className="text-gray-500">{s.quantity} shares</p>
//               </div>

//               <div className="text-right">
//                 <p className="text-gray-200 font-semibold">
//                   ₹{s.value.toLocaleString()}
//                 </p>
//                 <p className={color}>
//                   {s.pnlPercent >= 0 ? "+" : ""}
//                   {s.pnlPercent.toFixed(2)}%
//                 </p>
//               </div>
//             </div>
//           );
//         })}
//       </div>

//       {/* Button */}
//       <a
//         href="/portfolio"
//         className="block text-center rounded-lg py-3 bg-gray-800 border border-gray-700 
//         text-gray-200 hover:bg-gray-700 transition font-medium"
//       >
//         View Full Portfolio
//       </a>
//     </div>
//   );
// };

// export default PortfolioWidget;

"use client";

import { PortfolioHolding } from "@/types/stock";
import { Trash2, Briefcase } from "lucide-react";

interface PortfolioWidgetProps {
  portfolio: PortfolioHolding[];
  onDelete: (ticker: string) => void;
}

export default function PortfolioWidget({ portfolio, onDelete }: PortfolioWidgetProps) {
  if (!portfolio || portfolio.length === 0) {
    return (
      <div className="rounded-xl bg-[#0f1629] border border-[#1f2a40] p-6 text-center shadow-lg">
        <div className="flex items-center justify-center gap-2 text-gray-300">
          <Briefcase size={18} />
          <span className="text-lg font-semibold">Your Portfolio</span>
        </div>

        <p className="text-gray-500 mt-4">Your portfolio is empty.</p>

        <a
          href="/portfolio"
          className="mt-4 inline-block px-4 py-2 rounded-lg bg-gray-800 text-gray-200 hover:bg-gray-700 transition"
        >
          Open Portfolio
        </a>
      </div>
    );
  }

  const totalInvested = portfolio.reduce((sum, p) => sum + p.buyPrice * p.quantity, 0);
  const totalShares = portfolio.reduce((sum, p) => sum + p.quantity, 0);

  return (
    <div className="rounded-xl bg-[#0f1629] border border-[#1f2a40] p-6 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2 text-gray-200 font-semibold text-xl">
          <Briefcase size={20} />
          Your Portfolio
        </div>
      </div>

      <div className="mb-6">
        <div className="text-sm text-gray-400">Total Holdings</div>
        <div className="text-2xl font-semibold text-gray-100">{totalShares} shares</div>

        <div className="text-sm text-gray-400 mt-2">Total Invested</div>
        <div className="text-xl font-semibold text-gray-200">₹{totalInvested.toFixed(2)}</div>
      </div>

      <div className="space-y-3">
        {portfolio.map((stock, index) => (
          <div
            key={index}
            className="flex items-center justify-between bg-[#141e35] border border-[#25304a] p-3 rounded-lg"
          >
            <div>
              <div className="font-semibold text-gray-100">{stock.ticker}</div>
              <div className="text-sm text-gray-400">{stock.quantity} shares</div>
            </div>

            <button
              onClick={() => onDelete(stock.ticker)}
              className="text-red-400 hover:text-red-300 transition"
            >
              <Trash2 size={18} />
            </button>
          </div>
        ))}
      </div>

      <a
        href="/portfolio"
        className="block w-full text-center mt-6 py-3 rounded-xl bg-[#1a2337] hover:bg-[#24304a] text-gray-100 transition font-semibold"
      >
        View Full Portfolio
      </a>
    </div>
  );
}
