import { Link, useLocation } from "react-router-dom";
import { useEffect } from "react";
import { motion } from "framer-motion";
import { Home, TrendingUp } from "lucide-react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background text-foreground p-4">
      {/* Ambient bg */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[500px] h-[500px] bg-primary/5 rounded-full blur-[120px]" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center max-w-md"
      >
        {/* Animated 404 */}
        <div className="relative mb-8">
          <motion.span
            animate={{ y: [0, -10, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            className="text-8xl md:text-9xl font-black gradient-text block"
          >
            404
          </motion.span>
          <div className="absolute inset-0 text-8xl md:text-9xl font-black text-primary/5 blur-xl">
            404
          </div>
        </div>

        <h1 className="text-2xl font-bold text-foreground mb-3">Page Not Found</h1>
        <p className="text-muted-foreground mb-8 leading-relaxed">
          The page <code className="text-primary bg-primary/10 px-2 py-0.5 rounded text-sm">{location.pathname}</code> doesn't exist.
          It may have been moved or removed.
        </p>

        <Link
          to="/"
          className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-primary to-emerald-500
            text-primary-foreground font-semibold shadow-lg shadow-primary/20 hover:shadow-primary/30
            hover:scale-[1.02] active:scale-[0.98] transition-all duration-200"
          id="go-home-btn"
        >
          <Home className="w-4 h-4" />
          Back to Dashboard
        </Link>

        <div className="mt-12 flex items-center justify-center gap-2 text-muted-foreground/50 text-xs">
          <TrendingUp className="w-3 h-3" />
          Sent-Stock
        </div>
      </motion.div>
    </div>
  );
};

export default NotFound;
