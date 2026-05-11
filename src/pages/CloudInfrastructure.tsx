import { Header } from "@/components/Header";
import { 
  Cloud, 
  Cpu, 
  Database, 
  Globe, 
  Layers, 
  Zap, 
  Box, 
  ShieldCheck, 
  GitBranch, 
  Server,
  ArrowRight,
  ExternalLink
} from "lucide-react";
import { motion } from "framer-motion";

const CloudInfrastructure = () => {
  const services = [
    {
      title: "Amazon CloudFront",
      category: "Content Delivery",
      description: "Serves the React app from edge locations using Distribution E3FOMHOPSEVE4Y. Handles SSL termination and global caching.",
      icon: <Globe className="w-8 h-8 text-blue-400" />,
      stats: "ap-south-1 Edge",
      gradient: "from-blue-500/20 to-cyan-500/20",
      href: "https://console.aws.amazon.com/cloudfront/v3/home"
    },
    {
      title: "AWS S3",
      category: "Storage",
      description: "Hosts the static production bundle in bucket mpd-frontend-209757840945. Integrates with GitHub Actions for automated sync.",
      icon: <Layers className="w-8 h-8 text-orange-400" />,
      stats: "S3 Standard",
      gradient: "from-orange-500/20 to-yellow-500/20",
      href: "https://console.aws.amazon.com/s3/home?region=ap-south-1"
    },
    {
      title: "AWS ECS Fargate",
      category: "Compute",
      description: "Runs the FastAPI backend and FinBERT model in containers within mpd-cluster. Uses task-definition market-pulse-backend.",
      icon: <Box className="w-8 h-8 text-orange-500" />,
      stats: "Fargate v1.4.0",
      gradient: "from-orange-600/20 to-red-600/20",
      href: "https://console.aws.amazon.com/ecs/v2/clusters/mpd-cluster/services?region=ap-south-1"
    },
    {
      title: "Application Load Balancer",
      category: "Networking",
      description: "The entry point (mpd-backend-alb-1607763580) that routes traffic to ECS tasks. Monitors service health and manages SSL.",
      icon: <Server className="w-8 h-8 text-blue-500" />,
      stats: "Active State",
      gradient: "from-blue-600/20 to-indigo-600/20",
      href: "https://console.aws.amazon.com/ec2/v2/home?region=ap-south-1#LoadBalancers:"
    },
    {
      title: "Amazon RDS (PostgreSQL)",
      category: "Database",
      description: "Stores portfolio data in mpd-postgres-v2. Authenticates via Secrets Manager (mpd/backend/db-credentials-v2).",
      icon: <Database className="w-8 h-8 text-blue-600" />,
      stats: "db.t3.micro",
      gradient: "from-blue-700/20 to-purple-700/20",
      href: "https://console.aws.amazon.com/rds/home?region=ap-south-1#databases:"
    },
    {
      title: "Amazon ElastiCache",
      category: "Caching",
      description: "A Redis cluster (mpd-cache) used to store technical indicators and news sentiment to reduce API latency.",
      icon: <Zap className="w-8 h-8 text-red-500" />,
      stats: "cache.t3.micro",
      gradient: "from-red-500/20 to-pink-500/20",
      href: "https://console.aws.amazon.com/elasticache/home?region=ap-south-1#redis:"
    },
    {
      title: "AWS Lambda",
      category: "Serverless",
      description: "Managed via SAM. Specifically used for mpd-fetch-price and mpd-fetch-news to handle bursts in traffic efficiently.",
      icon: <Cpu className="w-8 h-8 text-yellow-500" />,
      stats: "Python 3.10",
      gradient: "from-yellow-500/20 to-orange-500/20",
      href: "https://console.aws.amazon.com/lambda/home?region=ap-south-1#/functions"
    },
    {
      title: "GitHub Actions",
      category: "CI/CD",
      description: "Automated workflows (backend-deploy.yml & frontend-deploy.yml) that build, test, and deploy to AWS on every push.",
      icon: <GitBranch className="w-8 h-8 text-white" />,
      stats: "2 Workflows",
      gradient: "from-gray-500/20 to-slate-500/20",
      href: "https://github.com/Akshad265k/Sent-Stock-ML-mini/actions"
    }
  ];

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      <Header />
      
      <main className="container mx-auto px-4 py-12">
        <div className="text-center mb-16">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-medium mb-4"
          >
            <Cloud className="w-3 h-3" />
            <span>Cloud-Native Architecture</span>
          </motion.div>
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-white to-gray-500 bg-clip-text text-transparent"
          >
            Market Pulse Infrastructure
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-muted-foreground text-lg max-w-2xl mx-auto"
          >
            A high-performance, distributed system built on AWS for institutional-grade financial analytics and AI-powered sentiment tracking.
          </motion.p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {services.map((service, index) => (
            <motion.a
              key={service.title}
              href={service.href}
              target="_blank"
              rel="noreferrer"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.1 }}
              className={`relative group p-6 rounded-2xl border border-white/10 bg-gradient-to-br ${service.gradient} backdrop-blur-md hover:border-primary/50 transition-all duration-300 hover:-translate-y-1 block`}
            >
              <div className="mb-4 flex justify-between items-start">
                {service.icon}
                <ExternalLink className="w-4 h-4 text-white/20 group-hover:text-primary transition-colors" />
              </div>
              <div className="text-xs font-bold text-primary mb-1 uppercase tracking-wider">
                {service.category}
              </div>
              <h3 className="text-xl font-bold mb-2 group-hover:text-primary transition-colors flex items-center gap-2">
                {service.title}
              </h3>
              <p className="text-sm text-gray-400 leading-relaxed mb-4">
                {service.description}
              </p>
              <div className="pt-4 border-t border-white/5 flex items-center justify-between">
                <span className="text-[10px] font-mono text-gray-500 uppercase">{service.stats}</span>
                <ShieldCheck className="w-4 h-4 text-green-500/50" />
              </div>
            </motion.a>
          ))}
        </div>

        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="mt-20 p-8 rounded-3xl border border-primary/20 bg-primary/5 relative overflow-hidden group"
        >
          <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-20 transition-opacity">
            <Cloud className="w-64 h-64 text-primary" />
          </div>
          
          <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-8">
            <div>
              <h2 className="text-3xl font-bold mb-4">Architecture Flow</h2>
              <p className="text-gray-400 max-w-xl mb-6">
                Requests flow through CloudFront to the Load Balancer, which scales the ECS backend dynamically. 
                Data-heavy tasks are offloaded to Lambda functions, while results are cached in Redis for instantaneous user experiences.
              </p>
              <div className="flex flex-wrap gap-4">
                <div className="flex items-center gap-2 text-sm bg-white/5 px-4 py-2 rounded-lg border border-white/10">
                  <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                  <span>Production Ready</span>
                </div>
                <div className="flex items-center gap-2 text-sm bg-white/5 px-4 py-2 rounded-lg border border-white/10">
                  <ShieldCheck className="w-4 h-4 text-primary" />
                  <span>AWS Well-Architected</span>
                </div>
              </div>
            </div>
            
            <div className="flex flex-col gap-3">
              <a 
                href="https://github.com/Akshad265k/Sent-Stock-ML-mini/actions" 
                target="_blank" 
                rel="noreferrer"
                className="flex items-center gap-2 px-6 py-3 rounded-xl bg-primary text-white font-bold hover:bg-primary/90 transition-all hover:scale-105 active:scale-95 shadow-lg shadow-primary/20"
              >
                <span>View CI/CD Pipeline</span>
                <GitBranch className="w-4 h-4" />
              </a>
              <a 
                href="https://ap-south-1.console.aws.amazon.com/" 
                target="_blank" 
                rel="noreferrer"
                className="flex items-center gap-2 px-6 py-3 rounded-xl bg-white/5 text-white font-medium hover:bg-white/10 transition-all border border-white/10"
              >
                <span>AWS Console</span>
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>
          </div>
        </motion.div>

        <footer className="mt-20 text-center text-gray-600 text-sm">
          <p>© 2026 Market Pulse Cloud • Build v1.0.4 • AWS Region: ap-south-1</p>
        </footer>
      </main>
    </div>
  );
};

export default CloudInfrastructure;
