// import { StockPrice } from "@/types/stock";
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
// import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

// interface StockChartProps {
//   prices: StockPrice[];
//   currentPrice: number;
//   predictionPrice?: number;
//   ticker: string;
// }

// export const StockChart = ({ prices, currentPrice, predictionPrice, ticker }: StockChartProps) => {
//   const chartData = prices.map((price) => ({
//     date: new Date(price.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
//     price: price.close,
//   }));

//   const minPrice = Math.min(...prices.map((p) => p.low)) * 0.98;
//   const maxPrice = Math.max(...prices.map((p) => p.high)) * 1.02;

//   return (
//     <Card>
//       <CardHeader>
//         <div className="flex items-center justify-between">
//           <CardTitle className="text-lg">{ticker} Price Chart</CardTitle>
//           <div className="text-right">
//             <div className="text-2xl font-bold mono-num">${currentPrice.toFixed(2)}</div>
//             <div className="text-xs text-muted-foreground">Current Price</div>
//           </div>
//         </div>
//       </CardHeader>
//       <CardContent>
//         <ResponsiveContainer width="100%" height={400}>
//           <LineChart data={chartData}>
//             <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
//             <XAxis 
//               dataKey="date" 
//               stroke="hsl(var(--muted-foreground))"
//               tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
//             />
//             <YAxis 
//               domain={[minPrice, maxPrice]}
//               stroke="hsl(var(--muted-foreground))"
//               tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
//               tickFormatter={(value) => `$${value.toFixed(0)}`}
//             />
//             <Tooltip
//               contentStyle={{
//                 backgroundColor: "hsl(var(--card))",
//                 border: "1px solid hsl(var(--border))",
//                 borderRadius: "8px",
//                 color: "hsl(var(--foreground))",
//               }}
//               formatter={(value: number) => [`$${value.toFixed(2)}`, "Price"]}
//             />
//             <Line
//               type="monotone"
//               dataKey="price"
//               stroke="hsl(var(--primary))"
//               strokeWidth={2}
//               dot={false}
//               activeDot={{ r: 4, fill: "hsl(var(--primary))" }}
//             />
//             {predictionPrice && (
//               <ReferenceLine
//                 y={predictionPrice}
//                 stroke="hsl(var(--accent))"
//                 strokeDasharray="5 5"
//                 strokeWidth={2}
//                 label={{
//                   value: `Target: $${predictionPrice.toFixed(2)}`,
//                   position: "right",
//                   fill: "hsl(var(--accent))",
//                   fontSize: 12,
//                 }}
//               />
//             )}
//           </LineChart>
//         </ResponsiveContainer>
//       </CardContent>
//     </Card>
//   );
// };
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface StockChartProps {
  prices: number[];
  currentPrice: number;
  predictionPrice: number;
  ticker: string;
}

export const StockChart = ({ prices, currentPrice, predictionPrice, ticker }: StockChartProps) => {
  if (!prices || prices.length === 0) {
    return (
      <div className="h-[300px] flex items-center justify-center border rounded-lg bg-gray-50">
        <p className="text-gray-400">No chart data available</p>
      </div>
    );
  }

  // Convert number[] → recharts format
  const data = prices.map((p, i) => ({
    day: `Day ${i + 1}`,
    price: p,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Price History ({ticker})</CardTitle>
      </CardHeader>

      <CardContent>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <XAxis dataKey="day" />
              <YAxis
                domain={["dataMin - 5", "dataMax + 5"]}
                tickFormatter={(v) => v.toFixed(0)}
              />
              <Tooltip />
              
              <Line
                type="monotone"
                dataKey="price"
                stroke="#2563eb"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};
