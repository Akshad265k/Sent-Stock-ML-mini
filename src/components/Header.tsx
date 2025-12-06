import { TrendingUp } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";

export const Header = () => {
  const location = useLocation();

  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 group">
          <div className="p-2 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
            <TrendingUp className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">Sent-Stock</h1>
            <p className="text-xs text-muted-foreground">AI-Powered Predictions</p>
          </div>
        </Link>

        <nav className="flex gap-6">
          <Link
            to="/"
            className={cn(
              "text-sm font-medium transition-colors hover:text-primary",
              location.pathname === "/" ? "text-primary" : "text-muted-foreground"
            )}
          >
            Dashboard
          </Link>
          <Link to="/portfolio" className={cn(
              "text-sm font-medium transition-colors hover:text-primary",
              location.pathname === "/portfolio" ? "text-primary" : "text-muted-foreground"
            )}>
            Portfolio
          </Link>
          <Link
            to="/about"
            className={cn(
              "text-sm font-medium transition-colors hover:text-primary",
              location.pathname === "/about" ? "text-primary" : "text-muted-foreground"
            )}
          >
            About
          </Link>
        </nav>
      </div>
    </header>
  );
};
