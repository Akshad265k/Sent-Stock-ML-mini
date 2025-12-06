import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { Prediction } from "@/types/stock";
import { Card, CardContent } from "@/components/ui/card";

interface PredictionHeroProps {
  prediction: Prediction;
  currentPrice: number;
}

export const PredictionHero = ({ prediction, currentPrice }: PredictionHeroProps) => {
  const getIcon = () => {
    switch (prediction.direction) {
      case "UP":
        return <TrendingUp className="w-16 h-16" />;
      case "DOWN":
        return <TrendingDown className="w-16 h-16" />;
      default:
        return <Minus className="w-16 h-16" />;
    }
  };

  const getColorClass = () => {
    switch (prediction.direction) {
      case "UP":
        return "text-success";
      case "DOWN":
        return "text-destructive";
      default:
        return "text-muted-foreground";
    }
  };

  const getBgClass = () => {
    switch (prediction.direction) {
      case "UP":
        return "bg-success/10";
      case "DOWN":
        return "bg-destructive/10";
      default:
        return "bg-muted/10";
    }
  };

  return (
    <Card className={`${getBgClass()} border-2 ${prediction.direction === "UP" ? "border-success/30" : prediction.direction === "DOWN" ? "border-destructive/30" : "border-border"}`}>
      <CardContent className="pt-6 text-center">
        <div className="mb-4">
          <span className="text-sm font-medium text-muted-foreground">AI Prediction</span>
        </div>
        
        <div className={`flex items-center justify-center mb-4 ${getColorClass()}`}>
          {getIcon()}
        </div>

        <h3 className={`text-4xl font-bold mb-2 ${getColorClass()}`}>
          {prediction.direction}
        </h3>

        <div className="space-y-2 text-foreground">
          {prediction.targetPrice && (
            <div className="flex justify-between items-center text-sm">
              <span className="text-muted-foreground">Target Price:</span>
              <span className="mono-num font-semibold">
                ${prediction.targetPrice.toFixed(2)}
              </span>
            </div>
          )}
          {prediction.change && (
            <div className="flex justify-between items-center text-sm">
              <span className="text-muted-foreground">Expected Change:</span>
              <span className={`mono-num font-semibold ${getColorClass()}`}>
                {prediction.change > 0 ? "+" : ""}{prediction.change.toFixed(2)}%
              </span>
            </div>
          )}
          <div className="flex justify-between items-center text-sm">
            <span className="text-muted-foreground">Confidence:</span>
            <span className="mono-num font-semibold">{prediction.confidence}%</span>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-border">
          <p className="text-xs text-muted-foreground">
            Prediction based on sentiment analysis and ML models
          </p>
        </div>
      </CardContent>
    </Card>
  );
};
