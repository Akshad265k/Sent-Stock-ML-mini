import { TrendingUp, TrendingDown } from "lucide-react";

interface StockHeaderProps {
  ticker: string;
  name: string;
  currentPrice: number;
  change: number;
  changePercent: number;
}

export const StockHeader = ({ ticker, name, currentPrice, change, changePercent }: StockHeaderProps) => {
  const isPositive = change >= 0;

  return (
    <div
      className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 p-6 rounded-2xl glass glow-primary animate-fade-up"
      id="stock-header"
    >
      <div>
        <div className="flex items-center gap-3">
          <h2 className="text-3xl font-bold gradient-text">{ticker}</h2>
          <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded-full border border-emerald-400/20">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-400"></span>
            </span>
            Live
          </span>
        </div>
        <p className="text-muted-foreground mt-1">{name}</p>
      </div>
      
      <div className="flex items-baseline gap-3">
        <span className="text-4xl font-bold mono-num text-foreground">
          ₹{currentPrice.toFixed(2)}
        </span>
        <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full ${
          isPositive
            ? "text-emerald-400 bg-emerald-400/10"
            : "text-rose-400 bg-rose-400/10"
        }`}>
          {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
          <span className="text-lg font-semibold mono-num">
            {isPositive ? "+" : ""}{change.toFixed(2)}
          </span>
          <span className="text-sm mono-num opacity-80">
            ({isPositive ? "+" : ""}{changePercent.toFixed(2)}%)
          </span>
        </div>
      </div>
    </div>
  );
};
