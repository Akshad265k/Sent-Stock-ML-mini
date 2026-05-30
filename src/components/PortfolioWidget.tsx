"use client";

import { PortfolioHolding } from "@/types/stock";
import { Trash2, Briefcase, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";

interface PortfolioWidgetProps {
  portfolio: PortfolioHolding[];
  onDelete: (ticker: string) => void;
}

export default function PortfolioWidget({ portfolio, onDelete }: PortfolioWidgetProps) {
  if (!portfolio || portfolio.length === 0) {
    return (
      <div className="rounded-2xl glass p-6 text-center" id="portfolio-widget-empty">
        <div className="flex items-center justify-center gap-2 text-foreground">
          <Briefcase size={18} className="text-primary" />
          <span className="text-lg font-semibold">Your Portfolio</span>
        </div>

        <p className="text-muted-foreground mt-4 text-sm">Your portfolio is empty.</p>

        <Link
          to="/portfolio"
          className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-secondary
            text-foreground hover:bg-secondary/80 transition-all text-sm font-medium"
        >
          Open Portfolio <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </div>
    );
  }

  const totalInvested = portfolio.reduce((sum, p) => sum + p.buyPrice * p.quantity, 0);
  const totalShares = portfolio.reduce((sum, p) => sum + p.quantity, 0);

  return (
    <div className="rounded-2xl glass p-6" id="portfolio-widget">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2 text-foreground font-semibold text-lg">
          <Briefcase size={18} className="text-primary" />
          Your Portfolio
        </div>
        <span className="text-xs text-muted-foreground bg-secondary/50 px-2 py-0.5 rounded-full">
          {portfolio.length} stocks
        </span>
      </div>

      <div className="mb-5 grid grid-cols-2 gap-4">
        <div>
          <div className="text-xs text-muted-foreground">Total Holdings</div>
          <div className="text-xl font-semibold text-foreground mono-num">{totalShares} shares</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Total Invested</div>
          <div className="text-xl font-semibold text-foreground mono-num">₹{totalInvested.toLocaleString()}</div>
        </div>
      </div>

      <div className="space-y-2">
        {portfolio.map((stock, index) => (
          <div
            key={index}
            className="flex items-center justify-between bg-secondary/30 border border-border/30 p-3 rounded-xl
              hover:bg-secondary/50 transition-colors group"
          >
            <div>
              <div className="font-semibold text-foreground text-sm">{stock.ticker}</div>
              <div className="text-xs text-muted-foreground mono-num">
                {stock.quantity} × ₹{stock.buyPrice.toFixed(2)}
              </div>
            </div>

            <button
              onClick={() => onDelete(stock.ticker)}
              className="text-muted-foreground hover:text-rose-400 transition-colors opacity-0
                group-hover:opacity-100 p-1.5 rounded-lg hover:bg-rose-400/10"
              aria-label={`Remove ${stock.ticker}`}
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>

      <Link
        to="/portfolio"
        className="flex items-center justify-center gap-2 w-full mt-5 py-3 rounded-xl
          bg-secondary/50 hover:bg-secondary text-foreground transition-all font-medium text-sm
          border border-border/30 hover:border-border/50"
      >
        View Full Portfolio <ArrowRight className="w-3.5 h-3.5" />
      </Link>
    </div>
  );
}
