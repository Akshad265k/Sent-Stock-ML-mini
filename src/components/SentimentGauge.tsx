import { StockSentiment } from "@/types/stock";

interface SentimentGaugeProps {
  sentiment: StockSentiment;
}

export const SentimentGauge = ({ sentiment }: SentimentGaugeProps) => {
  if (!sentiment) return null;

  const score = sentiment.score ?? 0;
  const label = sentiment.label || "Neutral";

  // Convert -1..1 → 0..1 for gauge
  const normalized = (score + 1) / 2;

  // SVG arc gauge parameters
  const radius = 60;
  const circumference = Math.PI * radius; // half-circle
  const gaugeOffset = circumference - (normalized * circumference);

  const getColor = () => {
    if (score > 0.3) return { text: "text-emerald-400", stroke: "hsl(142, 76%, 45%)", bg: "bg-emerald-400/10" };
    if (score < -0.3) return { text: "text-rose-400", stroke: "hsl(0, 72%, 51%)", bg: "bg-rose-400/10" };
    return { text: "text-amber-400", stroke: "hsl(45, 93%, 58%)", bg: "bg-amber-400/10" };
  };

  const colors = getColor();

  return (
    <div className="rounded-2xl glass p-6 animate-fade-up" id="sentiment-gauge">
      <h3 className="text-lg font-semibold text-foreground mb-6">News Sentiment</h3>

      <div className="flex flex-col items-center">
        {/* Arc Gauge */}
        <div className="relative w-40 h-24 mb-4">
          <svg className="w-40 h-24" viewBox="0 0 160 90" overflow="visible">
            {/* Background arc */}
            <path
              d="M 10 80 A 60 60 0 0 1 150 80"
              fill="none"
              stroke="hsl(220, 14%, 18%)"
              strokeWidth="8"
              strokeLinecap="round"
            />
            {/* Value arc */}
            <path
              d="M 10 80 A 60 60 0 0 1 150 80"
              fill="none"
              stroke={colors.stroke}
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={gaugeOffset}
              className="transition-all duration-1000 ease-out"
            />
            {/* Needle indicator dot */}
            {(() => {
              const angle = Math.PI - (normalized * Math.PI);
              const nx = 80 + radius * Math.cos(angle);
              const ny = 80 - radius * Math.sin(angle);
              return (
                <>
                  <circle cx={nx} cy={ny} r="6" fill={colors.stroke} className="drop-shadow-lg" />
                  <circle cx={nx} cy={ny} r="3" fill="hsl(220, 16%, 12%)" />
                </>
              );
            })()}
          </svg>

          {/* Center score */}
          <div className="absolute inset-0 flex items-end justify-center pb-0">
            <div className="text-center">
              <div className={`text-3xl font-bold mono-num ${colors.text}`}>
                {score.toFixed(2)}
              </div>
            </div>
          </div>
        </div>

        {/* Label badge */}
        <span className={`text-sm font-semibold px-4 py-1.5 rounded-full ${colors.bg} ${colors.text}`}>
          {label}
        </span>

        {/* Scale indicators */}
        <div className="flex justify-between w-full mt-4 text-[10px] text-muted-foreground uppercase tracking-wider">
          <span>Bearish</span>
          <span>Neutral</span>
          <span>Bullish</span>
        </div>
      </div>

      <div className="mt-5 pt-4 border-t border-border/50 text-center text-xs text-muted-foreground">
        Sentiment derived from AI analysis of recent news headlines
      </div>
    </div>
  );
};
