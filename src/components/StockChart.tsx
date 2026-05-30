import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceLine,
} from "recharts";

interface StockChartProps {
  prices: number[];
  currentPrice: number;
  predictionPrice: number;
  ticker: string;
}

export const StockChart = ({ prices, currentPrice, predictionPrice, ticker }: StockChartProps) => {
  if (!prices || prices.length === 0) {
    return (
      <div className="h-[350px] flex items-center justify-center rounded-2xl glass" id="chart-empty">
        <p className="text-muted-foreground">No chart data available</p>
      </div>
    );
  }

  // Convert number[] → recharts format
  const data = prices.map((p, i) => ({
    day: `Day ${i + 1}`,
    price: p,
  }));

  const minPrice = Math.min(...prices) * 0.98;
  const maxPrice = Math.max(...prices, predictionPrice || 0) * 1.02;

  return (
    <div className="rounded-2xl glass p-6 glow-primary animate-fade-up" id="stock-chart">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-foreground">
          Price History — <span className="gradient-text">{ticker}</span>
        </h3>
        <div className="text-right">
          <div className="text-2xl font-bold mono-num text-foreground">₹{currentPrice.toFixed(2)}</div>
          <div className="text-xs text-muted-foreground">Current Price</div>
        </div>
      </div>

      <div className="h-[300px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(189, 95%, 52%)" stopOpacity={0.3} />
                <stop offset="50%" stopColor="hsl(189, 95%, 52%)" stopOpacity={0.1} />
                <stop offset="100%" stopColor="hsl(189, 95%, 52%)" stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <XAxis
              dataKey="day"
              stroke="hsl(217, 10%, 40%)"
              tick={{ fill: "hsl(217, 10%, 50%)", fontSize: 11 }}
              axisLine={{ stroke: "hsl(220, 14%, 20%)" }}
              tickLine={false}
            />
            <YAxis
              domain={[minPrice, maxPrice]}
              stroke="hsl(217, 10%, 40%)"
              tick={{ fill: "hsl(217, 10%, 50%)", fontSize: 11 }}
              tickFormatter={(v) => `₹${v.toFixed(0)}`}
              axisLine={false}
              tickLine={false}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(220, 16%, 12%)",
                border: "1px solid hsl(220, 14%, 20%)",
                borderRadius: "12px",
                color: "hsl(210, 40%, 98%)",
                boxShadow: "0 10px 40px rgba(0,0,0,0.4)",
                padding: "12px 16px",
              }}
              formatter={(value: number) => [`₹${value.toFixed(2)}`, "Price"]}
              labelStyle={{ color: "hsl(217, 10%, 60%)", fontSize: 11 }}
            />

            {predictionPrice > 0 && (
              <ReferenceLine
                y={predictionPrice}
                stroke="hsl(45, 93%, 58%)"
                strokeDasharray="6 4"
                strokeWidth={2}
                label={{
                  value: `Target: ₹${predictionPrice.toFixed(0)}`,
                  position: "right",
                  fill: "hsl(45, 93%, 58%)",
                  fontSize: 11,
                  fontWeight: 600,
                }}
              />
            )}

            <Area
              type="monotone"
              dataKey="price"
              stroke="hsl(189, 95%, 52%)"
              strokeWidth={2.5}
              fill="url(#chartGradient)"
              dot={false}
              activeDot={{
                r: 5,
                fill: "hsl(189, 95%, 52%)",
                stroke: "hsl(220, 16%, 12%)",
                strokeWidth: 2,
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
