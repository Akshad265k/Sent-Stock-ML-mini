import { StockSentiment } from "@/types/stock";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface SentimentGaugeProps {
  sentiment: StockSentiment;
}

export const SentimentGauge = ({ sentiment }: SentimentGaugeProps) => {
  if (!sentiment) return null;

  // Backend: score = -1 to 1
  const score = sentiment.score ?? 0;

  // Convert -1..1 → 0..100
  const gaugeValue = ((score + 1) / 2) * 100;

  // Label from backend / fallback
  const label = sentiment.label || "Neutral";

  const getColor = () => {
    if (score > 0.3) return "text-emerald-600";
    if (score < -0.3) return "text-rose-600";
    return "text-gray-500";
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">News Sentiment</CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="text-center">
          <div className={`text-5xl font-bold mono-num mb-2 ${getColor()}`}>
            {score.toFixed(2)}
          </div>
          <div className={`text-sm font-medium ${getColor()}`}>
            {label}
          </div>
        </div>

        <div className="space-y-2">
          <Progress
            value={gaugeValue}
            className="h-3 bg-secondary"
            style={{
              // @ts-ignore
              "--progress-background": score > 0 ? "hsl(var(--success))" : "hsl(var(--destructive))",
            }}
          />
        </div>

        <div className="pt-4 border-t border-border text-center text-xs text-muted-foreground">
          Sentiment score derived from latest news headlines.
        </div>
      </CardContent>
    </Card>
  );
};
