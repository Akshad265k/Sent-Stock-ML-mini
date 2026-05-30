import { TrendingUp, Menu, X } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { useState, useEffect } from "react";

export const Header = () => {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const navLinks = [
    { to: "/", label: "Dashboard" },
    { to: "/portfolio", label: "Portfolio" },
    { to: "/about", label: "About" },
  ];

  // Show Cloud link only on localhost
  const isLocal = typeof window !== "undefined" &&
    ["localhost", "127.0.0.1"].includes(window.location.hostname);

  return (
    <header
      className={cn(
        "sticky top-0 z-50 transition-all duration-300",
        scrolled
          ? "glass-strong shadow-lg shadow-black/20"
          : "bg-card/30 backdrop-blur-sm border-b border-border/50"
      )}
    >
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-3 group" id="header-logo">
          <div className="relative p-2.5 rounded-xl bg-gradient-to-br from-primary/20 to-emerald-500/20 group-hover:from-primary/30 group-hover:to-emerald-500/30 transition-all duration-300 glow-primary">
            <TrendingUp className="w-6 h-6 text-primary" />
            <div className="absolute inset-0 rounded-xl bg-primary/10 blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground tracking-tight">
              Sent-Stock
            </h1>
            <div className="flex items-center gap-2">
              <p className="text-xs text-muted-foreground">AI-Powered Predictions</p>
              <span className="flex items-center gap-1 text-[10px] text-emerald-400 font-semibold uppercase tracking-wider">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-400"></span>
                </span>
                Live
              </span>
            </div>
          </div>
        </Link>

        {/* Desktop Nav */}
        <nav className="hidden md:flex items-center gap-1" id="desktop-nav">
          {navLinks.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              className={cn(
                "relative px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200",
                location.pathname === link.to
                  ? "text-primary bg-primary/10"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
              )}
            >
              {link.label}
              {location.pathname === link.to && (
                <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-4 h-0.5 bg-primary rounded-full" />
              )}
            </Link>
          ))}
          {isLocal && (
            <Link
              to="/infrastructure"
              className={cn(
                "text-sm font-medium transition-all duration-200 border border-primary/20 bg-primary/5 px-3 py-1.5 rounded-full ml-2",
                location.pathname === "/infrastructure"
                  ? "text-primary"
                  : "text-muted-foreground hover:text-primary hover:border-primary/40"
              )}
            >
              ☁️ Cloud
            </Link>
          )}
        </nav>

        {/* Mobile Hamburger */}
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="md:hidden p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-colors"
          id="mobile-menu-toggle"
          aria-label="Toggle navigation menu"
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile Nav Panel */}
      {mobileOpen && (
        <div className="md:hidden glass-strong border-t border-border/50 animate-fade-up" id="mobile-nav">
          <nav className="container mx-auto px-4 py-4 flex flex-col gap-1">
            {navLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                onClick={() => setMobileOpen(false)}
                className={cn(
                  "px-4 py-3 text-sm font-medium rounded-lg transition-all duration-200",
                  location.pathname === link.to
                    ? "text-primary bg-primary/10"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                )}
              >
                {link.label}
              </Link>
            ))}
            {isLocal && (
              <Link
                to="/infrastructure"
                onClick={() => setMobileOpen(false)}
                className="px-4 py-3 text-sm font-medium rounded-lg text-muted-foreground hover:text-primary hover:bg-secondary/50 transition-all"
              >
                ☁️ Cloud Infrastructure
              </Link>
            )}
          </nav>
        </div>
      )}
    </header>
  );
};
