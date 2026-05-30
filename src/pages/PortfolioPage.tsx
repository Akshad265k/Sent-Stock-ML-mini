"use client";

import { useEffect, useState } from "react";
import { PortfolioHolding, PortfolioResponse } from "@/types/stock";
import {
  TrendingUp,
  Wallet,
  Percent,
  ArrowUpRight,
  ArrowDownRight,
  Loader2,
} from "lucide-react";
import { Header } from "@/components/Header";
import { motion } from "framer-motion";

export default function PortfolioPage() {
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";
  const [portfolio, setPortfolio] = useState<PortfolioHolding[]>([]);
  const [stats, setStats] = useState<PortfolioResponse | null>(null);
  const [loading, setLoading] = useState(false);

  // Load holdings from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("ai_portfolio_holdings");
    if (saved) setPortfolio(JSON.parse(saved));
  }, []);

  // Fetch backend analysis
  useEffect(() => {
    if (portfolio.length === 0) return;

    const fetchStats = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API_BASE_URL}/portfolio/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ holdings: portfolio }),
        });
        if (!res.ok) return;
        const data = await res.json();
        setStats(data);
      } catch {
        // Silently fail
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, [portfolio]);

  if (!portfolio.length) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <Wallet className="w-12 h-12 text-muted-foreground/30 mb-4" />
          <h2 className="text-xl font-semibold text-foreground mb-2">Portfolio Empty</h2>
          <p className="text-muted-foreground max-w-sm">
            Analyze a stock on the dashboard and add it to your portfolio to see performance analytics here.
          </p>
        </div>
      </div>
    );
  }

  if (loading || !stats) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <div className="flex flex-col items-center justify-center py-24">
          <Loader2 className="w-8 h-8 text-primary animate-spin mb-4" />
          <p className="text-muted-foreground">Loading portfolio analysis...</p>
        </div>
      </div>
    );
  }

  const { overview, stocks } = stats;
  const isGain = overview.totalPnL >= 0;
  const gainColor = isGain ? "text-emerald-400" : "text-rose-400";

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header />

      {/* Ambient bg */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-primary/5 rounded-full blur-[120px]" />
      </div>

      <main className="container mx-auto px-4 py-8 space-y-8">
        {/* Title */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <Wallet className="w-7 h-7 text-primary" />
            <span className="gradient-text">Portfolio</span>
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Track your stock holdings and performance
          </p>
        </motion.div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            {
              label: "Total Value",
              value: `₹${overview.currentValue.toLocaleString()}`,
              icon: Wallet,
              color: "text-foreground",
            },
            {
              label: "Total Gain/Loss",
              value: `${isGain ? "+" : ""}₹${overview.totalPnL.toLocaleString()}`,
              icon: isGain ? ArrowUpRight : ArrowDownRight,
              color: gainColor,
            },
            {
              label: "Return",
              value: `${isGain ? "+" : ""}${overview.totalPnLPercent}%`,
              icon: Percent,
              color: gainColor,
            },
          ].map((card, i) => (
            <motion.div
              key={card.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: i * 0.1 }}
              className="glass rounded-2xl p-6"
              id={`summary-${card.label.toLowerCase().replace(/\s/g, "-")}`}
            >
              <div className="flex items-center justify-between mb-2">
                <p className="text-muted-foreground text-sm">{card.label}</p>
                <card.icon className={`w-5 h-5 ${card.color} opacity-60`} />
              </div>
              <p className={`text-3xl font-bold mono-num ${card.color}`}>
                {card.value}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Holdings Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.3 }}
          className="glass rounded-2xl p-6 shadow-lg"
          id="holdings-table"
        >
          <h2 className="text-xl font-semibold mb-6 text-foreground">Holdings</h2>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="text-muted-foreground border-b border-border/50 text-xs uppercase tracking-wider">
                  <th className="py-3 pr-4">Stock</th>
                  <th className="py-3 pr-4">Shares</th>
                  <th className="py-3 pr-4">Avg Price</th>
                  <th className="py-3 pr-4">Current Price</th>
                  <th className="py-3 pr-4">Market Value</th>
                  <th className="py-3">Gain/Loss</th>
                </tr>
              </thead>

              <tbody>
                {stocks.map((s, i) => {
                  const color = s.pnl >= 0 ? "text-emerald-400" : "text-rose-400";
                  return (
                    <motion.tr
                      key={s.ticker}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3, delay: 0.4 + i * 0.05 }}
                      className="border-b border-border/30 hover:bg-secondary/20 transition-colors"
                    >
                      <td className="py-4 pr-4">
                        <div className="font-semibold text-foreground">{s.ticker}</div>
                        <div className="text-xs text-muted-foreground">{s.sentiment.label}</div>
                      </td>
                      <td className="py-4 pr-4 mono-num">{s.quantity}</td>
                      <td className="py-4 pr-4 mono-num">₹{s.buyPrice.toLocaleString()}</td>
                      <td className="py-4 pr-4 mono-num">₹{s.currentPrice.toLocaleString()}</td>
                      <td className="py-4 pr-4 font-semibold mono-num">
                        ₹{s.value.toLocaleString()}
                      </td>
                      <td className={`py-4 font-semibold mono-num ${color}`}>
                        {s.pnl >= 0 ? "+" : ""}₹{s.pnl.toLocaleString()}
                        <div className="text-xs">
                          ({s.pnlPercent >= 0 ? "+" : ""}{s.pnlPercent}%)
                        </div>
                      </td>
                    </motion.tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
