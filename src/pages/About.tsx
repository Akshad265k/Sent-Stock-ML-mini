import { Header } from "@/components/Header";
import { AlertTriangle, Brain, TrendingUp, Newspaper, Code2, Cloud, Database } from "lucide-react";
import { motion } from "framer-motion";

const About = () => {
  const fadeUp = {
    initial: { opacity: 0, y: 20 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true },
    transition: { duration: 0.5 },
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header />

      {/* Ambient bg */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-1/4 w-[500px] h-[500px] bg-primary/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-20 right-1/4 w-[400px] h-[400px] bg-emerald-500/5 rounded-full blur-[100px]" />
      </div>

      <main className="container mx-auto px-4 py-12 max-w-4xl">
        {/* Hero */}
        <motion.div {...fadeUp} className="mb-12 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium mb-4">
            <Brain className="w-3.5 h-3.5" />
            About the Project
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            About <span className="gradient-text">Sent-Stock</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            AI-powered stock predictions driven by news sentiment analysis for Indian markets
          </p>
        </motion.div>

        <div className="space-y-6">
          {/* Disclaimer */}
          <motion.div {...fadeUp} className="rounded-2xl border border-rose-500/30 bg-rose-500/5 p-6" id="disclaimer-card">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-rose-500/10">
                <AlertTriangle className="w-5 h-5 text-rose-400" />
              </div>
              <h2 className="text-lg font-semibold text-rose-400">Important Disclaimer</h2>
            </div>
            <div className="space-y-2 text-muted-foreground text-sm leading-relaxed">
              <p>
                This is a <strong className="text-foreground">research and educational project</strong>.
                All predictions are for informational purposes only — this is NOT financial advice.
              </p>
              <p>
                Past performance and sentiment analysis do not guarantee future results.
                Always consult a qualified financial advisor before making investment decisions.
              </p>
            </div>
          </motion.div>

          {/* How It Works */}
          <motion.div {...fadeUp} className="rounded-2xl glass p-6" id="how-it-works">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded-lg bg-primary/10">
                <Brain className="w-5 h-5 text-primary" />
              </div>
              <h2 className="text-lg font-semibold text-foreground">How It Works</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                {
                  icon: Newspaper,
                  step: "1",
                  title: "News Collection",
                  desc: "Continuously monitors business news from major financial sources including Reuters, Bloomberg, CNBC, and more.",
                },
                {
                  icon: Brain,
                  step: "2",
                  title: "Sentiment Analysis",
                  desc: "Advanced NLP and machine learning models analyze each article for sentiment scores (-1 to +1).",
                },
                {
                  icon: TrendingUp,
                  step: "3",
                  title: "Prediction Generation",
                  desc: "ML models combine sentiment with historical price patterns to generate trend predictions with confidence scores.",
                },
              ].map((item, i) => (
                <motion.div
                  key={item.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: i * 0.1 }}
                  className="relative p-5 rounded-xl bg-secondary/30 border border-border/30 hover:border-primary/30 transition-all duration-300"
                >
                  <div className="absolute -top-3 -left-2 w-7 h-7 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center text-xs font-bold text-primary">
                    {item.step}
                  </div>
                  <item.icon className="w-6 h-6 text-primary mb-3" />
                  <h3 className="font-semibold text-foreground text-sm mb-2">{item.title}</h3>
                  <p className="text-xs text-muted-foreground leading-relaxed">{item.desc}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Tech Stack */}
          <motion.div {...fadeUp} className="rounded-2xl glass p-6" id="tech-stack">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded-lg bg-primary/10">
                <Code2 className="w-5 h-5 text-primary" />
              </div>
              <h2 className="text-lg font-semibold text-foreground">Technology Stack</h2>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { name: "React + TypeScript", icon: "⚛️" },
                { name: "Vite", icon: "⚡" },
                { name: "Tailwind CSS", icon: "🎨" },
                { name: "Recharts", icon: "📊" },
                { name: "Python / FastAPI", icon: "🐍" },
                { name: "NLP / Transformers", icon: "🧠" },
                { name: "AWS (ECS, S3, ALB)", icon: "☁️" },
                { name: "Framer Motion", icon: "✨" },
              ].map((tech) => (
                <div
                  key={tech.name}
                  className="p-3 rounded-xl bg-secondary/30 border border-border/30 text-center hover:border-primary/30 transition-colors"
                >
                  <span className="text-2xl block mb-2">{tech.icon}</span>
                  <span className="text-xs font-medium text-muted-foreground">{tech.name}</span>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Methodology */}
          <motion.div {...fadeUp} className="rounded-2xl glass p-6" id="methodology">
            <h2 className="text-lg font-semibold text-foreground mb-4">Methodology</h2>
            <p className="text-muted-foreground text-sm mb-4">
              Our prediction models use a combination of:
            </p>
            <ul className="space-y-2">
              {[
                "Real-time news sentiment analysis using transformer-based NLP models",
                "Historical stock price data and technical indicators",
                "Volume and volatility analysis",
                "Weighted sentiment aggregation from multiple sources",
                "Time-series forecasting models (LSTM, GRU)",
              ].map((item) => (
                <li key={item} className="flex items-start gap-3 text-sm text-muted-foreground">
                  <span className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                  {item}
                </li>
              ))}
            </ul>
          </motion.div>

          {/* R&D */}
          <motion.div {...fadeUp} className="rounded-2xl glass p-6" id="research">
            <h2 className="text-lg font-semibold text-foreground mb-4">Research & Development</h2>
            <p className="text-muted-foreground text-sm mb-4 leading-relaxed">
              This project demonstrates the potential of combining AI, NLP, and financial data to create
              predictive models for educational purposes. Future enhancements include:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                { icon: Database, text: "Multi-modal analysis (social media, SEC filings)" },
                { icon: TrendingUp, text: "Longer-term prediction horizons" },
                { icon: Brain, text: "Portfolio optimization suggestions" },
                { icon: Cloud, text: "Real-time alert systems" },
              ].map((item) => (
                <div key={item.text} className="flex items-center gap-3 p-3 rounded-lg bg-secondary/20 text-sm text-muted-foreground">
                  <item.icon className="w-4 h-4 text-primary shrink-0" />
                  {item.text}
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default About;
