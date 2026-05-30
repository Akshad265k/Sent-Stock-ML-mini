import { useState } from "react";
import { Search, Loader2, Sparkles } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

interface StockSearchProps {
  onSearch: (ticker: string) => void;
  currentTicker?: string;
  isLoading?: boolean;
}

const POPULAR_TICKERS = [
  { symbol: "RELIANCE", label: "Reliance" },
  { symbol: "TCS", label: "TCS" },
  { symbol: "INFY", label: "Infosys" },
  { symbol: "TATAMOTORS", label: "Tata Motors" },
  { symbol: "HDFCBANK", label: "HDFC Bank" },
  { symbol: "ITC", label: "ITC" },
];

export const StockSearch = ({ onSearch, currentTicker, isLoading = false }: StockSearchProps) => {
  const [ticker, setTicker] = useState(currentTicker || "");
  const [isFocused, setIsFocused] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (ticker.trim()) {
      onSearch(ticker.trim().toUpperCase());
    }
  };

  return (
    <div className="space-y-4" id="stock-search">
      <form onSubmit={handleSubmit} className="relative flex gap-2 w-full max-w-lg mx-auto">
        {/* Glow ring when focused */}
        <div
          className={`absolute -inset-1 rounded-xl bg-gradient-to-r from-primary/30 via-emerald-500/30 to-primary/30 blur-lg transition-opacity duration-500 ${
            isFocused ? "opacity-100" : "opacity-0"
          }`}
        />

        <div className="relative flex-1">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground z-10" />
          <Input
            placeholder="Enter ticker (e.g. TATAMOTORS)"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            className="pl-9 relative bg-card/80 border-border/60 focus:border-primary/50 transition-all duration-300"
            disabled={isLoading}
            id="search-input"
          />
        </div>

        <Button
          type="submit"
          disabled={isLoading}
          className="relative bg-gradient-to-r from-primary to-emerald-500 hover:from-primary/90 hover:to-emerald-500/90 text-primary-foreground font-semibold shadow-lg shadow-primary/20 transition-all duration-300 hover:shadow-primary/30 hover:scale-[1.02] active:scale-[0.98]"
          id="search-button"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Analyzing
            </>
          ) : (
            <>
              <Sparkles className="mr-2 h-4 w-4" />
              Analyze
            </>
          )}
        </Button>
      </form>

      {/* Popular tickers */}
      <div className="flex items-center justify-center gap-2 flex-wrap">
        <span className="text-xs text-muted-foreground font-medium">Popular:</span>
        {POPULAR_TICKERS.map((t) => (
          <button
            key={t.symbol}
            onClick={() => {
              setTicker(t.symbol);
              onSearch(t.symbol);
            }}
            disabled={isLoading}
            className={`text-xs px-3 py-1.5 rounded-full transition-all duration-200 font-medium ${
              currentTicker?.includes(t.symbol)
                ? "bg-primary/20 text-primary border border-primary/30"
                : "bg-secondary/50 hover:bg-secondary text-muted-foreground hover:text-foreground border border-transparent hover:border-border/50"
            } disabled:opacity-50`}
            id={`ticker-pill-${t.symbol}`}
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  );
};