// import React from "react";
// import { StockPrediction } from "@/types/stock";
// import { ArrowUpRight, ArrowDownRight, Target, AlertCircle, TrendingUp } from "lucide-react";

// interface PredictionCardProps {
//   currentPrice: number;
//   prediction: StockPrediction;
// }

// export const PredictionCard = ({ currentPrice, prediction }: PredictionCardProps) => {
//   // Safety: if backend fails
//   if (!currentPrice) return null;

//   // SAFE destructuring with defaults
//   const {
//     targetPrice = currentPrice,
//     confidence = 0.5,
//     timeframe = "7 Days",
//     signal = "NEUTRAL",
//   } = prediction || {};

//   const potential =
//     currentPrice > 0 ? ((targetPrice - currentPrice) / currentPrice) * 100 : 0;

//   const isPositive = potential > 0;

//   // Safe signal
//   const safeSignal = signal.toUpperCase();

//   let colorClass = "text-gray-500";
//   let bgClass = "bg-gray-100";
//   let borderClass = "border-gray-200";
//   let Icon = AlertCircle;

//   if (safeSignal.includes("BUY")) {
//     colorClass = "text-emerald-600";
//     bgClass = "bg-emerald-50";
//     borderClass = "border-emerald-200";
//     Icon = ArrowUpRight;
//   } else if (safeSignal.includes("SELL")) {
//     colorClass = "text-rose-600";
//     bgClass = "bg-rose-50";
//     borderClass = "border-rose-200";
//     Icon = ArrowDownRight;
//   }

//   const progressWidth = Math.min(Math.abs(potential) * 5, 100);

//   return (
//     <div className={`rounded-xl border-2 ${borderClass} shadow-sm overflow-hidden bg-white`}>
//       <div className={`${bgClass} border-b ${borderClass} p-6`}>
//         <div className="flex justify-between items-start">
//           <div>
//             <p className="text-sm font-medium text-gray-500 uppercase tracking-wider">
//               AI Verdict ({timeframe})
//             </p>
//             <h2 className={`text-3xl font-bold ${colorClass} flex items-center gap-2 mt-1`}>
//               {safeSignal}
//               <Icon className="h-8 w-8" />
//             </h2>
//           </div>
//           <div className="text-right">
//             <div className="text-sm font-medium text-gray-500">Confidence</div>
//             <div className="text-2xl font-bold text-gray-900">{(confidence * 100).toFixed(0)}%</div>
//           </div>
//         </div>
//       </div>

//       <div className="p-6 space-y-6">
//         <div className="grid grid-cols-2 gap-4">
//           <div className="space-y-1">
//             <span className="text-sm text-gray-500">Current Price</span>
//             <div className="text-2xl font-semibold text-gray-900">₹{currentPrice.toFixed(2)}</div>
//           </div>
//           <div className="space-y-1 text-right">
//             <span className="text-sm text-gray-500 flex items-center justify-end gap-1">
//               <Target className="h-4 w-4" /> Target Price
//             </span>
//             <div className={`text-2xl font-bold ${colorClass}`}>
//               ₹{targetPrice.toFixed(2)}
//             </div>
//           </div>
//         </div>

//         <div className="space-y-2">
//           <div className="flex justify-between text-sm">
//             <span className="text-gray-500">Potential Return</span>
//             <span className={`font-bold ${isPositive ? "text-emerald-600" : "text-rose-600"}`}>
//               {potential > 0 ? "+" : ""}
//               {potential.toFixed(2)}%
//             </span>
//           </div>

//           <div className="h-3 w-full bg-gray-100 rounded-full overflow-hidden">
//             <div
//               className={`h-full transition-all duration-500 ${
//                 isPositive ? "bg-emerald-500" : "bg-rose-500"
//               }`}
//               style={{ width: `${progressWidth}%` }}
//             />
//           </div>
//         </div>

//         <div className="bg-gray-50 rounded-lg p-4 text-sm text-gray-600 flex gap-2 items-start border border-gray-100">
//           <TrendingUp className="h-4 w-4 mt-0.5 shrink-0 text-gray-400" />
//           <p>
//             Analysis based on technical indicators (RSI, MA-50) and sentiment from recent news articles.
//           </p>
//         </div>
//       </div>
//     </div>
//   );
// };


import React from "react";
import { StockPrediction } from "@/types/stock";
import { ArrowUpRight, ArrowDownRight, Minus, Target, TrendingUp } from "lucide-react";

interface PredictionCardProps {
  currentPrice: number;
  prediction: StockPrediction;
}

export const PredictionCard = ({ currentPrice, prediction }: PredictionCardProps) => {
  if (!currentPrice) return null;

  const {
    targetPrice = currentPrice,
    confidence = 0.5,
    timeframe = "7 Days",
    signal = "HOLD",
  } = prediction || {};

  const potential =
    currentPrice > 0 ? ((targetPrice - currentPrice) / currentPrice) * 100 : 0;

  const isPositive = potential > 0;
  const safeSignal = signal.toUpperCase();

  // --- SIGNAL STYLE CONFIG ---
  const signalStyles = {
    BUY: {
      color: "text-emerald-400",
      bg: "bg-emerald-900/20",
      border: "border-emerald-700",
      glow: "shadow-[0_0_15px_rgba(16,185,129,0.25)]",
      icon: ArrowUpRight,
    },
    SELL: {
      color: "text-rose-400",
      bg: "bg-rose-900/20",
      border: "border-rose-700",
      glow: "shadow-[0_0_15px_rgba(244,63,94,0.25)]",
      icon: ArrowDownRight,
    },
    HOLD: {
      color: "text-gray-400",
      bg: "bg-gray-800/40",
      border: "border-gray-600",
      glow: "shadow-[0_0_10px_rgba(255,255,255,0.08)]",
      icon: Minus,
    },
  }[safeSignal];

  const Icon = signalStyles.icon;

  const progressColor = isPositive
    ? "from-emerald-500 to-emerald-300"
    : "from-rose-500 to-rose-300";

  return (
    <div
      className={`rounded-2xl border ${signalStyles.border} ${signalStyles.bg} ${signalStyles.glow}
      shadow-lg backdrop-blur-xl transition-all duration-300`}
    >
      {/* Header */}
      <div className="p-6 border-b border-gray-700/60">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-widest font-medium text-gray-400">
              AI VERDICT • {timeframe}
            </p>

            <h2 className={`text-4xl font-extrabold mt-1 flex items-center gap-2 ${signalStyles.color}`}>
              {safeSignal}
              <Icon className="w-8 h-8" />
            </h2>
          </div>

          <div className="text-right">
            <p className="text-xs uppercase tracking-wide text-gray-400">Confidence</p>
            <p className="text-3xl font-bold text-gray-100">{(confidence * 100).toFixed(0)}%</p>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="p-6 space-y-8">
        
        {/* Prices */}
        <div className="grid grid-cols-2 gap-6">
          <div>
            <p className="text-xs text-gray-500">Current Price</p>
            <p className="text-3xl font-semibold text-gray-100">₹{currentPrice.toFixed(2)}</p>
          </div>

          <div className="text-right">
            <p className="text-xs text-gray-500 flex items-center justify-end gap-2">
              <Target className="h-4 w-4" /> Target Price
            </p>
            <p className={`text-3xl font-bold ${signalStyles.color}`}>
              ₹{targetPrice.toFixed(2)}
            </p>
          </div>
        </div>

        {/* Potential Return */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Potential Return</span>
            <span className={`font-bold ${isPositive ? "text-emerald-400" : "text-rose-400"}`}>
              {potential > 0 ? "+" : ""}
              {potential.toFixed(2)}%
            </span>
          </div>

          {/* Gradient Progress */}
          <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full bg-gradient-to-r ${progressColor}`}
              style={{ width: `${Math.min(Math.abs(potential) * 4, 100)}%` }}
            />
          </div>
        </div>

        {/* Info Box */}
        <div className="bg-gray-900/40 rounded-xl p-4 border border-gray-700 text-sm flex gap-3">
          <TrendingUp className="w-5 h-5 text-gray-500 mt-0.5" />
          <p className="text-gray-400 leading-relaxed">
            Prediction generated using technical indicators (RSI, MA-50) combined with AI-powered news sentiment.
          </p>
        </div>

      </div>
    </div>
  );
};
