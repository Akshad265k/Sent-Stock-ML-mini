// import { NewsArticle } from "@/types/stock";
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
// import { ScrollArea } from "@/components/ui/scroll-area";
// import { Clock } from "lucide-react";

// interface NewsFeedProps {
//   news: NewsArticle[];
// }

// export const NewsFeed = ({ news }: NewsFeedProps) => {
//   const getSentimentColor = (sentiment: number) => {
//     if (sentiment > 0.3) return "border-l-success bg-success/5";
//     if (sentiment < -0.3) return "border-l-destructive bg-destructive/5";
//     return "border-l-accent bg-accent/5";
//   };

//   const getSentimentBadge = (sentiment: number) => {
//     if (sentiment > 0.3) return { label: "Positive", color: "bg-success/20 text-success" };
//     if (sentiment < -0.3) return { label: "Negative", color: "bg-destructive/20 text-destructive" };
//     return { label: "Neutral", color: "bg-accent/20 text-accent" };
//   };

//   return (
//     <Card>
//       <CardHeader>
//         <CardTitle className="text-lg">Live News Feed</CardTitle>
//       </CardHeader>
//       <CardContent>
//         <ScrollArea className="h-[500px] pr-4">
//           <div className="space-y-4">
//             {news.map((article) => {
//               const badge = getSentimentBadge(article.sentiment);
//               return (
//                 <div
//                   key={article.id}
//                   className={`p-4 rounded-lg border-l-4 transition-colors hover:bg-secondary/50 ${getSentimentColor(article.sentiment)}`}
//                 >
//                   <div className="flex items-start justify-between gap-3 mb-2">
//                     <h4 className="font-medium text-foreground leading-tight flex-1">
//                       {article.headline}
//                     </h4>
//                     <span className={`text-xs px-2 py-1 rounded-full whitespace-nowrap ${badge.color}`}>
//                       {badge.label}
//                     </span>
//                   </div>
//                   <div className="flex items-center gap-3 text-xs text-muted-foreground">
//                     <span className="font-medium">{article.source}</span>
//                     <span className="flex items-center gap-1">
//                       <Clock className="w-3 h-3" />
//                       {article.publishedAt}
//                     </span>
//                   </div>
//                 </div>
//               );
//             })}
//           </div>
//         </ScrollArea>
//       </CardContent>
//     </Card>
//   );
// // };
// import { StockNewsItem } from "@/types/stock";
// import { ExternalLink, Newspaper } from "lucide-react";
// import React from "react";

// interface NewsFeedProps {
//   news: StockNewsItem[];
// }

// export const NewsFeed = ({ news }: NewsFeedProps) => {
//   const getSentimentStyles = (sentiment: string = "Neutral") => {
//     const safeSentiment = sentiment.toLowerCase();

//     if (safeSentiment.includes("positive")) {
//       return {
//         border: "border-l-emerald-500",
//         bg: "bg-emerald-50/50",
//         badge: "bg-emerald-100 text-emerald-700",
//         label: "Positive",
//       };
//     }
//     if (safeSentiment.includes("negative")) {
//       return {
//         border: "border-l-rose-500",
//         bg: "bg-rose-50/50",
//         badge: "bg-rose-100 text-rose-700",
//         label: "Negative",
//       };
//     }
//     return {
//       border: "border-l-gray-300",
//       bg: "bg-gray-50/50",
//       badge: "bg-gray-100 text-gray-700",
//       label: "Neutral",
//     };
//   };

//   return (
//     <div className="rounded-xl border shadow-sm bg-white overflow-hidden">
//       <div className="p-4 border-b bg-gray-50 flex items-center gap-2">
//         <Newspaper className="h-5 w-5 text-gray-500" />
//         <h3 className="font-semibold text-lg text-gray-900">Live News Feed</h3>
//       </div>

//       <div className="max-h-[500px] overflow-y-auto p-4 space-y-3 custom-scrollbar">
//         {news && news.length > 0 ? (
//           news.map((article, index) => {
//             const styles = getSentimentStyles(article.sentiment || "Neutral");

//             return (
//               <div
//                 key={index}
//                 className={`p-4 rounded-r-lg border-l-4 transition-all hover:bg-gray-50 ${styles.border} ${styles.bg}`}
//               >
//                 <div className="flex items-start justify-between gap-3 mb-2">
//                   <a
//                     href={article.url}
//                     target="_blank"
//                     rel="noopener noreferrer"
//                     className="font-medium text-gray-900 hover:text-blue-600 hover:underline leading-tight flex-1 flex items-start gap-1 group"
//                   >
//                     {article.title}
//                     <ExternalLink className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity mt-1 text-blue-500" />
//                   </a>

//                   <span className={`text-xs px-2.5 py-0.5 rounded-full font-medium whitespace-nowrap ${styles.badge}`}>
//                     {styles.label}
//                   </span>
//                 </div>

//                 <div className="flex items-center gap-3 text-xs text-gray-500">
//                   <span className="font-semibold uppercase tracking-wide text-xs text-gray-400">
//                     {article.source || "Unknown Source"}
//                   </span>
//                 </div>
//               </div>
//             );
//           })
//         ) : (
//           <div className="text-center py-10 text-gray-400">
//             <p>No recent news found for this stock.</p>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// };

import { StockNewsItem } from "@/types/stock";
import { ExternalLink, Newspaper } from "lucide-react";
import React from "react";

interface NewsFeedProps {
  news: StockNewsItem[];
}

export const NewsFeed = ({ news }: NewsFeedProps) => {
  const getSentimentStyles = (sentiment: string = "Neutral") => {
    const safeSentiment = sentiment.toLowerCase();

    if (safeSentiment.includes("positive")) {
      return {
        border: "border-l-emerald-500/90",
        bg: "bg-emerald-900/10",
        badge: "bg-emerald-700/30 text-emerald-300 border border-emerald-600/40",
        label: "Positive",
      };
    }
    if (safeSentiment.includes("negative")) {
      return {
        border: "border-l-rose-500/90",
        bg: "bg-rose-900/10",
        badge: "bg-rose-700/30 text-rose-300 border border-rose-600/40",
        label: "Negative",
      };
    }
    return {
      border: "border-l-blue-400/90",
      bg: "bg-gray-900/10",
      badge: "bg-gray-700/30 text-gray-300 border border-gray-600/40",
      label: "Neutral",
    };
  };

  return (
    <div className="rounded-2xl border border-gray-800 bg-gray-900/60 backdrop-blur-xl shadow-xl overflow-hidden">
      
      {/* Header */}
      <div className="p-4 border-b border-gray-800 flex items-center gap-2 bg-gray-900/40">
        <Newspaper className="h-5 w-5 text-gray-400" />
        <h3 className="font-semibold text-lg text-gray-200 tracking-wide">
          Live News Feed
        </h3>
      </div>

      <div className="max-h-[500px] overflow-y-auto p-4 space-y-3 custom-scrollbar">
        {news && news.length > 0 ? (
          news.map((article, index) => {
            const styles = getSentimentStyles(article.sentiment || "Neutral");

            return (
              <div
                key={index}
                className={`p-4 rounded-xl border-l-4 ${styles.border} ${styles.bg}
                  transition-all duration-300 hover:bg-gray-800/40 shadow-md`}
              >
                {/* Title + sentiment */}
                <div className="flex items-start justify-between gap-4">
                  <a
                    href={article.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-medium text-gray-200 hover:text-blue-400 hover:underline 
                      leading-snug flex-1 flex items-start gap-1 group"
                  >
                    {article.title}
                    <ExternalLink
                      className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity 
                      mt-1 text-blue-400"
                    />
                  </a>

                  <span
                    className={`text-xs px-2.5 py-0.5 rounded-full font-medium 
                      whitespace-nowrap ${styles.badge}`}
                  >
                    {styles.label}
                  </span>
                </div>

                {/* Source Line */}
                <div className="flex items-center gap-3 text-xs text-gray-500 mt-2">
                  <span className="font-semibold uppercase tracking-wide text-gray-400">
                    {article.source || "Unknown Source"}
                  </span>
                </div>
              </div>
            );
          })
        ) : (
          <div className="text-center py-10 text-gray-500">
            <p>No recent news found for this stock.</p>
          </div>
        )}
      </div>
    </div>
  );
};
