import { StockNewsItem } from "@/types/stock";
import { ExternalLink, Newspaper } from "lucide-react";
import { motion } from "framer-motion";

interface NewsFeedProps {
  news: StockNewsItem[];
}

export const NewsFeed = ({ news }: NewsFeedProps) => {
  const getSentimentStyles = (sentiment: string = "Neutral") => {
    const s = sentiment.toLowerCase();
    if (s.includes("positive")) {
      return {
        border: "border-l-emerald-500/80",
        bg: "hover:bg-emerald-900/10",
        badge: "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25",
        label: "Positive",
      };
    }
    if (s.includes("negative")) {
      return {
        border: "border-l-rose-500/80",
        bg: "hover:bg-rose-900/10",
        badge: "bg-rose-500/15 text-rose-400 border border-rose-500/25",
        label: "Negative",
      };
    }
    return {
      border: "border-l-blue-400/60",
      bg: "hover:bg-blue-900/5",
      badge: "bg-gray-500/15 text-gray-400 border border-gray-500/20",
      label: "Neutral",
    };
  };

  return (
    <div className="rounded-2xl glass overflow-hidden animate-fade-up" id="news-feed">
      {/* Header */}
      <div className="p-4 border-b border-border/50 flex items-center gap-2">
        <Newspaper className="h-5 w-5 text-primary/70" />
        <h3 className="font-semibold text-lg text-foreground tracking-tight">
          Live News Feed
        </h3>
        {news && news.length > 0 && (
          <span className="text-xs text-muted-foreground ml-auto">
            {news.length} articles
          </span>
        )}
      </div>

      <div className="max-h-[500px] overflow-y-auto p-4 space-y-2 custom-scrollbar">
        {news && news.length > 0 ? (
          news.map((article, index) => {
            const styles = getSentimentStyles(article.sentiment || "Neutral");

            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                className={`p-4 rounded-xl border-l-4 ${styles.border} ${styles.bg}
                  transition-all duration-200 group`}
              >
                {/* Title + sentiment */}
                <div className="flex items-start justify-between gap-4">
                  <a
                    href={article.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-medium text-foreground/90 hover:text-primary leading-snug
                      flex-1 flex items-start gap-1 group/link transition-colors duration-200"
                  >
                    {article.title}
                    <ExternalLink className="h-3 w-3 opacity-0 group-hover/link:opacity-100
                      transition-opacity mt-1 text-primary shrink-0" />
                  </a>

                  <span className={`text-[10px] px-2.5 py-0.5 rounded-full font-semibold
                    whitespace-nowrap uppercase tracking-wider ${styles.badge}`}>
                    {styles.label}
                  </span>
                </div>

                {/* Source */}
                <div className="flex items-center gap-3 text-xs text-muted-foreground mt-2">
                  <span className="font-medium uppercase tracking-wide">
                    {article.source || "Unknown"}
                  </span>
                </div>
              </motion.div>
            );
          })
        ) : (
          <div className="text-center py-16 text-muted-foreground">
            <Newspaper className="h-8 w-8 mx-auto mb-3 opacity-30" />
            <p>No recent news found for this stock.</p>
          </div>
        )}
      </div>
    </div>
  );
};
