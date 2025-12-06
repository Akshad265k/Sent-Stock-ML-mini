// import { useState } from "react";
// import { Search } from "lucide-react";
// import { Input } from "@/components/ui/input";
// import { Button } from "@/components/ui/button";

// interface StockSearchProps {
//   onSearch: (ticker: string) => void;
//   currentTicker?: string;
// }

// export const StockSearch = ({ onSearch, currentTicker }: StockSearchProps) => {
//   const [ticker, setTicker] = useState(currentTicker || "");

//   const handleSubmit = (e: React.FormEvent) => {
//     e.preventDefault();
//     if (ticker.trim()) {
//       onSearch(ticker.toUpperCase());
//     }
//   };

//   const popularTickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"];

//   return (
//     <div className="space-y-4">
//       <form onSubmit={handleSubmit} className="flex gap-2">
//         <div className="relative flex-1">
//           <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
//           <Input
//             type="text"
//             placeholder="Enter stock ticker (e.g., AAPL)"
//             value={ticker}
//             onChange={(e) => setTicker(e.target.value.toUpperCase())}
//             className="pl-10 bg-secondary border-border"
//           />
//         </div>
//         <Button type="submit" className="bg-primary hover:bg-primary/90">
//           Search
//         </Button>
//       </form>

//       <div className="flex flex-wrap gap-2">
//         <span className="text-sm text-muted-foreground">Popular:</span>
//         {popularTickers.map((t) => (
//           <button
//             key={t}
//             onClick={() => {
//               setTicker(t);
//               onSearch(t);
//             }}
//             className={cn(
//               "text-xs px-3 py-1 rounded-full transition-colors",
//               currentTicker === t
//                 ? "bg-primary text-primary-foreground"
//                 : "bg-secondary hover:bg-secondary/80 text-foreground"
//             )}
//           >
//             {t}
//           </button>
//         ))}
//       </div>
//     </div>
//   );
// };

// function cn(...classes: (string | boolean | undefined)[]) {
//   return classes.filter(Boolean).join(" ");
// }


import { useState } from "react";
import { Search, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

interface StockSearchProps {
  onSearch: (ticker: string) => void;
  currentTicker?: string;
  isLoading?: boolean; // ✅ Added this definition
}

export const StockSearch = ({ onSearch, currentTicker, isLoading = false }: StockSearchProps) => {
  const [ticker, setTicker] = useState(currentTicker || "");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (ticker.trim()) {
      onSearch(ticker.trim().toUpperCase());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 w-full max-w-md mx-auto">
      <div className="relative flex-1">
        <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Enter ticker (e.g. TATAMOTORS)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          className="pl-9"
          disabled={isLoading}
        />
      </div>
      <Button type="submit" disabled={isLoading}>
        {isLoading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Searching
          </>
        ) : (
          "Analyze"
        )}
      </Button>
    </form>
  );
};