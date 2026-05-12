# 📈 Market Pulse: AI-Powered Financial Intelligence Platform

## Project Report

**Author:** Akshad Khune  
**Project Title:** Market Pulse Dashboard (Sent-Stock)  
**Date:** May 2026

---

## 1. Executive Summary
Market Pulse is a high-performance, cloud-native web application designed to empower investors with AI-driven insights. By fusing technical stock data with real-time sentiment analysis of financial news using the **FinBERT** transformer model, the platform provides a holistic "AI Verdict" for stocks. The project is fully deployed on **Amazon Web Services (AWS)** using a serverless-first approach (Fargate, S3, CloudFront) and features a completely automated **CI/CD pipeline** via GitHub Actions.

---

## 2. Problem Statement & Motivation
In modern financial markets, investors are overwhelmed by two separate streams of information:
1.  **Technical Data:** Numerical trends like price, RSI, and moving averages.
2.  **Qualitative Data:** News headlines, earnings reports, and global sentiment.

Most retail tools focus on one or the other. Market Pulse was built to bridge this gap, providing a unified dashboard that "reads" the news and analyzes the charts simultaneously to give a clear BUY, SELL, or HOLD recommendation based on evidence from both domains.

---

## 3. Technical Architecture

### 3.1 Frontend (The User Experience)
- **Framework:** React 18 with TypeScript.
- **Build Tool:** Vite (for near-instant HMR).
- **Styling:** Vanilla CSS & Tailwind CSS with a "Glassmorphism" dark-themed aesthetic.
- **Components:** Shadcn UI & Framer Motion (for smooth micro-animations).
- **Visualization:** Recharts for dynamic, interactive stock price charts.

### 3.2 Backend (The Engine)
- **Framework:** FastAPI (Python).
- **Server:** Uvicorn (ASGI).
- **ORM:** Peewee (connecting to PostgreSQL and SQLite).
- **Ticker Resolution:** `yahooquery` for intelligent stock search and symbol matching.
- **Market Data:** `yfinance` for historical price fetching.

### 3.3 AI Layer (Sentiment Analysis)
- **Model:** `yiyanghkust/finbert-tone`.
- **Logic:** 
    1.  Fetch top 5-10 news headlines via `GNews`.
    2.  Process headlines through the FinBERT pipeline.
    3.  Aggregate scores into a sentiment gauge (Positive, Neutral, Negative).
    4.  Combine sentiment with RSI/MA50 indicators to produce the final AI Verdict.

---

## 4. Cloud Infrastructure (AWS)
The project is built using a **Well-Architected** cloud infrastructure in the `ap-south-1` (Mumbai) region.

| Service | Role | Implementation Detail |
| :--- | :--- | :--- |
| **Amazon CloudFront** | CDN & SSL | Distribution `E3FOMHOPSEVE4Y` provides HTTPS and global edge delivery. |
| **Amazon S3** | Static Hosting | Bucket `mpd-frontend-209757840945` stores the compiled React app. |
| **AWS ECS Fargate** | Compute | Runs containerized FastAPI backend in `mpd-cluster`. Serverless—no servers to manage. |
| **Application Load Balancer** | Networking | Entry point `mpd-backend-alb-1607763580` routes traffic and manages health checks. |
| **Amazon RDS** | Database | Managed PostgreSQL instance `mpd-postgres-v2` for portfolio persistence. |
| **Secrets Manager** | Security | Stores DB credentials (`mpd/backend/db-credentials-v2`) to prevent hardcoding. |
| **SSM Parameter Store** | Config | Stores the live API URL for the frontend build process. |

---

## 5. CI/CD & DevOps Automation
The project implements "Infrastructure as Code" (CloudFormation) and a zero-touch deployment pipeline using **GitHub Actions**:

- **Frontend Workflow:** On push to `master`, it builds the React app, syncs it to S3, and invalidates the CloudFront cache.
- **Backend Workflow:** On push, it builds a new Docker image, pushes it to **Amazon ECR**, and performs a rolling update on the ECS service.

This ensures **zero-downtime** deployments and that the "Cloud" is always in sync with the latest code.

---

## 6. Key Features
- **Intelligent Search:** Search by company name (e.g., "Reliance" or "Apple") and get the correct ticker resolved automatically.
- **AI Verdict Card:** A clear summary of Target Price, Confidence Score, and AI Signal (BUY/SELL/HOLD).
- **Interactive Technicals:** View RSI and MA50 trends overlaid on price action.
- **News Sentiment Feed:** Real-time headlines categorized by AI sentiment.
- **Portfolio Tracker:** Add stocks to a personal portfolio to track gains/losses (persisted in RDS).
- **Cloud Showcase Page:** A built-in private page to demonstrate the cloud architecture.

---

## 7. Conclusion & Future Scope
Market Pulse successfully demonstrates how AI and Cloud Computing can be integrated to create a professional-grade financial tool. By automating the entire infrastructure and deployment process, the project remains scalable and maintainable.

**Future Enhancements:**
- Integration of **AWS ElastiCache (Redis)** for sub-millisecond price caching.
- Implementation of **AWS Lambda** for offloading news-scraping tasks to save ECS costs.
- Real-time **WebSocket** support for live price ticks.

---

## 8. References
1.  *Araci, D. (2019).* FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.
2.  *Bollen, J., et al. (2011).* Twitter mood predicts the stock market.
3.  *AWS Documentation:* Elastic Container Service and CloudFront Best Practices.
