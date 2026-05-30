import { StockPrediction } from "@/types/stock";
import { ArrowUpRight, ArrowDownRight, Minus, Target, TrendingUp } from "lucide-react";
import { motion } from "framer-motion";

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

  const signalConfig: Record<string, { color: string; bg: string; border: string; glow: string; icon: typeof ArrowUpRight }> = {
    BUY: {
      color: "text-emerald-400",
      bg: "bg-emerald-900/20",
      border: "border-emerald-700/50",
      glow: "glow-success",
      icon: ArrowUpRight,
    },
    SELL: {
      color: "text-rose-400",
      bg: "bg-rose-900/20",
      border: "border-rose-700/50",
      glow: "glow-danger",
      icon: ArrowDownRight,
    },
    HOLD: {
      color: "text-gray-400",
      bg: "bg-gray-800/40",
      border: "border-gray-600/50",
      glow: "",
      icon: Minus,
    },
  };

  const styles = signalConfig[safeSignal] || signalConfig.HOLD;
  const Icon = styles.icon;

  const progressColor = isPositive
    ? "from-emerald-500 to-emerald-300"
    : "from-rose-500 to-rose-300";

  // Circular confidence gauge
  const circumference = 2 * Math.PI * 40;
  const gaugeOffset = circumference - (confidence * circumference);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className={`rounded-2xl border ${styles.border} ${styles.bg} ${styles.glow}
      shadow-lg backdrop-blur-xl transition-all duration-300`}
      id="prediction-card"
    >
      {/* Header */}
      <div className="p-6 border-b border-gray-700/40">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-widest font-medium text-muted-foreground">
              AI VERDICT • {timeframe}
            </p>
            <h2 className={`text-4xl font-extrabold mt-1 flex items-center gap-2 ${styles.color}`}>
              {safeSignal}
              <Icon className="w-8 h-8" />
            </h2>
          </div>

          {/* Circular Confidence Gauge */}
          <div className="relative w-20 h-20">
            <svg className="w-20 h-20 -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50" cy="50" r="40"
                fill="none"
                stroke="hsl(220, 14%, 20%)"
                strokeWidth="6"
              />
              <circle
                cx="50" cy="50" r="40"
                fill="none"
                stroke="hsl(189, 95%, 52%)"
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={gaugeOffset}
                className="transition-all duration-1000 ease-out"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-lg font-bold text-foreground">{(confidence * 100).toFixed(0)}%</span>
              <span className="text-[9px] text-muted-foreground uppercase tracking-wider">Conf</span>
            </div>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="p-6 space-y-6">
        {/* Prices */}
        <div className="grid grid-cols-2 gap-6">
          <div>
            <p className="text-xs text-muted-foreground mb-1">Current Price</p>
            <p className="text-2xl font-semibold text-foreground mono-num">₹{currentPrice.toFixed(2)}</p>
          </div>
          <div className="text-right">
            <p className="text-xs text-muted-foreground mb-1 flex items-center justify-end gap-1.5">
              <Target className="h-3.5 w-3.5" /> Target Price
            </p>
            <p className={`text-2xl font-bold mono-num ${styles.color}`}>
              ₹{targetPrice.toFixed(2)}
            </p>
          </div>
        </div>

        {/* Potential Return */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Potential Return</span>
            <span className={`font-bold mono-num ${isPositive ? "text-emerald-400" : "text-rose-400"}`}>
              {potential > 0 ? "+" : ""}
              {potential.toFixed(2)}%
            </span>
          </div>
          <div className="w-full h-2.5 bg-secondary rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(Math.abs(potential) * 4, 100)}%` }}
              transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
              className={`h-full bg-gradient-to-r ${progressColor} rounded-full`}
            />
          </div>
        </div>

        {/* Info Box */}
        <div className="glass rounded-xl p-4 text-sm flex gap-3">
          <TrendingUp className="w-5 h-5 text-muted-foreground mt-0.5 shrink-0" />
          <p className="text-muted-foreground leading-relaxed">
            Prediction generated using technical indicators (RSI, MA-50) combined with AI-powered news sentiment analysis.
          </p>
        </div>
      </div>
    </motion.div>
  );
};
