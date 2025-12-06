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
    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
      <div>
        <h2 className="text-3xl font-bold text-foreground">{ticker}</h2>
        <p className="text-muted-foreground">{name}</p>
      </div>
      
      <div className="flex items-baseline gap-3">
        <span className="text-4xl font-bold mono-num text-foreground">
          ${currentPrice.toFixed(2)}
        </span>
        <div className={`flex items-center gap-1 ${isPositive ? "text-success" : "text-destructive"}`}>
          {isPositive ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
          <span className="text-xl font-semibold mono-num">
            {isPositive ? "+" : ""}{change.toFixed(2)}
          </span>
          <span className="text-lg mono-num">
            ({isPositive ? "+" : ""}{changePercent.toFixed(2)}%)
          </span>
        </div>
      </div>
    </div>
  );
};
