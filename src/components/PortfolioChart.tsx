"use client";

import { useEffect, useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";
import { PortfolioHolding } from "@/types/stock";

interface Props {
  portfolio: PortfolioHolding[];
}

export default function PortfolioChart({ portfolio }: Props) {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    if (portfolio.length === 0) return;

    const fetchHistory = async () => {
      // Step 1: call backend to analyze portfolio
      const res = await fetch("http://localhost:8000/api/portfolio/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ holdings: portfolio }),
      });

      if (!res.ok) return;

      const stats = await res.json();

      // Step 2: Generate fake value history for now
      // We simulate 10 days of growth based on final value

      const endValue = stats.overview.currentValue;
      const startValue = endValue * 0.95; // assume slow growth

      const points = [];

      for (let i = 0; i < 10; i++) {
        const value = startValue + ((endValue - startValue) * i) / 9;
        points.push({
          day: `Day ${i + 1}`,
          value: Number(value.toFixed(2)),
        });
      }

      setData(points);
    };

    fetchHistory();
  }, [portfolio]);

  if (portfolio.length === 0)
    return (
      <div className="rounded-xl bg-gray-900/60 border border-gray-800 p-6 text-center text-gray-400">
        No portfolio data to show.
      </div>
    );

  return (
    <div className="rounded-xl bg-gray-900/60 border border-gray-800 p-6 shadow-lg">
      <h2 className="text-xl font-semibold mb-4 text-gray-100">
        Portfolio Performance
      </h2>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <Line type="monotone" dataKey="value" stroke="#00C6FF" strokeWidth={3} dot={false} />
            <XAxis dataKey="day" stroke="#666" />
            <YAxis stroke="#666" />
            <Tooltip
              contentStyle={{
                background: "#111827",
                border: "1px solid #374151",
                borderRadius: "8px",
                color: "#fff",
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
