import { Header } from "@/components/Header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, Brain, TrendingUp, Newspaper } from "lucide-react";

const About = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      <main className="container mx-auto px-4 py-12 max-w-4xl">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4">About Sent-Stock</h1>
          <p className="text-xl text-muted-foreground">
            AI-powered stock predictions driven by news sentiment analysis
          </p>
        </div>

        <div className="space-y-6">
          <Card className="border-destructive/50">
            <CardHeader>
              <div className="flex items-center gap-3">
                <AlertTriangle className="w-6 h-6 text-destructive" />
                <CardTitle className="text-destructive">Important Disclaimer</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-foreground font-medium">
                This is a research and educational project.
              </p>
              <p className="text-muted-foreground">
                All predictions and analyses provided on this website are for <strong>informational and educational purposes only</strong>. 
                This is NOT financial advice. Do not make investment decisions based solely on the predictions shown here.
              </p>
              <p className="text-muted-foreground">
                Past performance and sentiment analysis do not guarantee future results. Always consult with a qualified financial advisor 
                before making any investment decisions. The creators of this tool are not responsible for any financial losses incurred.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-primary" />
                How It Works
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h4 className="font-semibold flex items-center gap-2">
                  <Newspaper className="w-4 h-4 text-primary" />
                  1. News Collection
                </h4>
                <p className="text-muted-foreground text-sm">
                  Our system continuously monitors and collects business news from major financial sources including 
                  Reuters, Bloomberg, CNBC, Wall Street Journal, and more.
                </p>
              </div>

              <div className="space-y-2">
                <h4 className="font-semibold flex items-center gap-2">
                  <Brain className="w-4 h-4 text-primary" />
                  2. Sentiment Analysis
                </h4>
                <p className="text-muted-foreground text-sm">
                  Advanced Natural Language Processing (NLP) and machine learning models analyze each article to determine 
                  sentiment scores ranging from -1 (very negative) to +1 (very positive).
                </p>
              </div>

              <div className="space-y-2">
                <h4 className="font-semibold flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-primary" />
                  3. Prediction Generation
                </h4>
                <p className="text-muted-foreground text-sm">
                  Machine learning models combine sentiment data with historical price patterns to generate short-term 
                  trend predictions (UP, DOWN, or NEUTRAL) with confidence scores.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Methodology</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-muted-foreground">
                Our prediction models use a combination of:
              </p>
              <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                <li>Real-time news sentiment analysis using transformer-based NLP models</li>
                <li>Historical stock price data and technical indicators</li>
                <li>Volume and volatility analysis</li>
                <li>Weighted sentiment aggregation from multiple sources</li>
                <li>Time-series forecasting models (LSTM, GRU)</li>
              </ul>
              <p className="text-muted-foreground text-sm pt-2">
                The models are continuously trained and updated to improve accuracy. However, market prediction is 
                inherently uncertain and multiple factors beyond news sentiment influence stock prices.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Research & Development</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                This project demonstrates the potential of combining AI, NLP, and financial data to create predictive 
                models for educational purposes. It serves as a showcase of modern machine learning techniques applied 
                to the financial domain.
              </p>
              <p className="text-muted-foreground mt-3">
                The system is designed to be extensible, allowing for future enhancements such as:
              </p>
              <ul className="list-disc list-inside space-y-1 text-muted-foreground mt-2">
                <li>Multi-modal analysis (social media, earnings calls, SEC filings)</li>
                <li>Longer-term prediction horizons</li>
                <li>Portfolio optimization suggestions</li>
                <li>Real-time alert systems</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default About;
