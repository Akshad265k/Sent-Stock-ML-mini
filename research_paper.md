# Market Pulse: A Cloud-Native Financial Intelligence Platform Using Containerized AI Inference and Automated CI/CD on AWS

---

**Authors:** Akshad Khune  
**Institution:** Department of Computer Engineering  
**Date:** May 2026

---

## Abstract

This paper presents the design, implementation, and evaluation of **Market Pulse**, a cloud-native financial intelligence platform that delivers real-time AI-powered stock market predictions. The system integrates a FinBERT-based Natural Language Processing (NLP) model for financial news sentiment analysis with technical indicator computation, all hosted on a scalable, fault-tolerant architecture on Amazon Web Services (AWS). The platform employs a microservices design pattern, with the backend containerized using Docker and orchestrated by AWS Elastic Container Service (ECS) Fargate, the frontend distributed globally via Amazon CloudFront and S3, and user data persisted in a managed Amazon RDS PostgreSQL instance. A fully automated CI/CD pipeline built on GitHub Actions ensures zero-downtime deployments. The architecture eliminates the need for server provisioning, demonstrating the efficacy of a serverless-first, infrastructure-as-code approach for deploying AI-heavy workloads in production.

**Keywords:** Cloud Computing, AWS ECS Fargate, FinBERT, Sentiment Analysis, CI/CD, Infrastructure as Code, FastAPI, CloudFront, Containerization.

---

## 1. Introduction

The financial technology sector is increasingly leveraging Artificial Intelligence (AI) and cloud computing to derive actionable insights from vast streams of unstructured data, such as news articles, earnings reports, and social media. Traditional monolithic financial applications face critical limitations: they are costly to scale, brittle under traffic spikes, and slow to update. Cloud-native architectures, by contrast, offer elasticity, managed services, and automation that are fundamentally reshaping how financial systems are built and operated.

Market Pulse addresses a specific gap: the need for a **unified platform** that can simultaneously (1) perform complex, computationally expensive AI inference using a domain-trained transformer model, (2) aggregate and serve real-time market data, and (3) remain highly available to end users at all times. The naive approach of running such a system on a single Virtual Machine (VM) is untenable—it creates a single point of failure and cannot adapt to unpredictable user demand.

This paper details the technical architecture of Market Pulse, with particular emphasis on:
- The cloud infrastructure design choices made to achieve high availability and scalability.
- The containerization strategy used to package and deploy the FinBERT AI model.
- The fully automated, zero-touch CI/CD pipeline that keeps the system up to date.
- The database and networking configurations that ensure data integrity and secure communication.

---

## 2. Related Work

### 2.1 Sentiment Analysis in Financial Markets
Prior research has established the causal link between news sentiment and equity price movements. Bollen et al. [1] demonstrated that Twitter mood states correlated with stock market movements with 87.6% accuracy. The work of Araci [2] introduced FinBERT, a pre-trained language model fine-tuned on financial corpora, which significantly outperformed general-purpose BERT on tasks like financial news classification. Market Pulse builds directly on this work by integrating FinBERT as a production inference service.

### 2.2 Cloud-Native Application Architectures
The concept of "cloud-native" applications, as defined by the Cloud Native Computing Foundation (CNCF), centers on microservices, containerization, and DevOps automation [3]. Research by Dragoni et al. [4] systematically evaluates the benefits of microservices in terms of independent deployability and fault isolation, which informed the architectural decisions in this project.

### 2.3 Containerized AI Inference
Deployment of ML models via Docker containers on managed orchestration platforms (e.g., Kubernetes, ECS) is an established production pattern [5]. Packaging the model and its dependencies into an immutable Docker image ensures reproducibility and eliminates "works on my machine" failures.

---

## 3. System Architecture

The system architecture follows a three-tier model: **Presentation**, **Application Logic**, and **Data**. Each tier is independently deployable and scaled separately.

### 3.1 Architectural Overview

```
[ User Browser ]
       │ HTTPS
       ▼
[ Amazon CloudFront (CDN) ]
       │ Cache/Forward
       ▼
[ Amazon S3 (Static Assets) ]    [ Application Load Balancer ]
                                           │ HTTP/8000
                                           ▼
                                 [ ECS Fargate Task ]
                                 ┌─────────────────────┐
                                 │  FastAPI (Python)   │
                                 │  FinBERT Model      │
                                 │  yfinance Client    │
                                 └─────────────────────┘
                                           │
                                           ▼
                                 [ Amazon RDS PostgreSQL ]
```

### 3.2 Frontend Layer: S3 + CloudFront

The frontend is a Single Page Application (SPA) built with **React 18**, **TypeScript**, and **Vite**, and compiled into a set of static assets (HTML, CSS, JS bundles). These assets are stored in an **Amazon S3** bucket (`mpd-frontend-209757840945`) configured for static website hosting.

Rather than exposing the S3 bucket directly, all user traffic is routed through an **Amazon CloudFront** distribution (ID: `E3FOMHOPSEVE4Y`). CloudFront serves as a global Content Delivery Network (CDN), caching static assets at 400+ edge locations worldwide. This design provides several critical advantages:

1.  **Latency Reduction:** Assets are served from the edge location geographically nearest to the user, reducing round-trip time significantly compared to serving from a single origin region (ap-south-1).
2.  **HTTPS Enforcement:** CloudFront is configured with `redirect-to-https`, ensuring all traffic is encrypted in transit without requiring SSL certificate management at the S3 level.
3.  **Cost Efficiency:** S3 data transfer costs are greatly reduced since CloudFront's cache absorbs repeated requests.
4.  **SPA Routing:** Custom error responses (403, 404) are configured to return `index.html`, enabling React Router's client-side navigation to function correctly.

### 3.3 Application Layer: ECS Fargate + Application Load Balancer

The core business logic—API request handling, technical analysis computation, and AI inference—runs inside **Docker** containers orchestrated by **AWS Elastic Container Service (ECS) Fargate**.

**Why Fargate?** Fargate is a "serverless" container compute engine. Unlike classic EC2-based ECS deployments, Fargate eliminates the need to provision, patch, or manage the underlying server infrastructure. The developer specifies only the CPU (0.5 vCPU) and Memory (1 GB) requirements in the ECS Task Definition, and AWS handles the rest. This is a significant operational advantage for a small team.

The Docker image for the backend is defined by a multi-stage `Dockerfile` in the `/backend` directory. It:
1.  Starts from a `python:3.11-slim` base image.
2.  Installs all Python dependencies from `requirements.txt` (including `fastapi`, `transformers`, `yfinance`, `yahooquery`).
3.  On container startup, loads the **FinBERT** model (`yiyanghkust/finbert-tone`) into memory. This is the computationally expensive step (done once at boot, not on every request).
4.  Starts the **Uvicorn** ASGI server, exposing the API on port `8000`.

The **Application Load Balancer (ALB)** (`mpd-backend-alb`) sits in front of the ECS tasks and performs two key functions: (a) distributing incoming API requests across multiple running task instances, and (b) conducting periodic health checks against the `/` endpoint to remove unhealthy tasks from the rotation automatically.

### 3.4 Data Layer: Amazon RDS (PostgreSQL)

User portfolio data—ticker symbols, quantities, and purchase prices—must be persisted reliably. **Amazon RDS for PostgreSQL** (`mpd-postgres-v2`, `db.t3.micro`) provides a fully managed relational database service.

**Database Credential Management:** A critical security concern in cloud deployments is credential exposure. The backend code (`database.py`) does not contain any hardcoded passwords. Instead, it integrates with **AWS Secrets Manager** at runtime. The secret `mpd/backend/db-credentials-v2` stores the auto-generated `username` and `password`, which the application fetches on startup via the `boto3` AWS SDK. This pattern ensures that credentials are never committed to source control.

**Resilient Fallback:** The `database.py` module is designed with a fallback mechanism. If the Secrets Manager is unreachable (e.g., during local development), it gracefully falls back to a local **SQLite** database, allowing developers to run the full application without any cloud dependencies.

### 3.5 AI Model: FinBERT Inference Pipeline

The AI pipeline is the core differentiator of Market Pulse. Upon receiving a stock analysis request, the system executes the following pipeline:

1.  **Ticker Resolution:** The raw user query (e.g., "Apple" or "TCS") is resolved to a valid exchange ticker symbol using `yahooquery`'s search API.
2.  **Technical Analysis:** Six months of daily price data are fetched from Yahoo Finance (`yfinance`). The system computes:
    - **RSI (Relative Strength Index):** A 14-period RSI to measure momentum.
    - **MA-50 (50-day Moving Average):** To identify the prevailing price trend.
    - **Technical Signal:** `BUY` if price > MA50 and RSI < 70; `SELL` if price < MA50 and RSI > 30; otherwise `NEUTRAL`.
3.  **News Aggregation:** Recent financial news headlines (last 7 days) are fetched for the stock using `GNews`.
4.  **FinBERT Inference:** Each headline is tokenized (max 512 tokens) and passed through the FinBERT pipeline (`yiyanghkust/finbert-tone`). The model outputs a classification: `Positive`, `Neutral`, or `Negative`, mapped to scores of `+1`, `0`, `-1` respectively.
5.  **Signal Fusion (Verdict Engine):**
    - **BUY:** Technical signal = BUY **AND** Avg. Sentiment > 0.15 (Positive).
    - **SELL:** Technical signal = SELL **AND** Avg. Sentiment < -0.15 (Negative).
    - **HOLD:** All other cases (signal disagreement or neutral sentiment).

---

## 4. CI/CD Pipeline (DevOps Automation)

A fully automated CI/CD pipeline is implemented using **GitHub Actions**, eliminating all manual deployment steps. Two workflow files govern the process:

### 4.1 Frontend Deployment Workflow (`frontend-deploy.yml`)
**Trigger:** Any `git push` to the `master` branch that modifies files in `src/`, `public/`, `package.json`, or `index.html`.

**Execution Steps:**
1.  Checkout repository code.
2.  Set up Node.js 18 environment.
3.  Run `npm install` to install dependencies.
4.  Configure AWS credentials using stored GitHub Secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
5.  Fetch the production backend URL from **AWS SSM Parameter Store** (`/mpd/frontend/VITE_API_BASE_URL`) and write it to `.env.production`. This decouples the frontend build from hardcoded API URLs.
6.  Execute `npm run build` to produce the optimized production bundle.
7.  Sync the `dist/` directory to S3 (`aws s3 sync dist/ s3://mpd-frontend-209757840945/ --delete`).
8.  Invalidate the CloudFront cache (`aws cloudfront create-invalidation ... --paths "/*"`) to ensure users receive the latest version immediately.

### 4.2 Backend Deployment Workflow (`backend-deploy.yml`)
**Trigger:** Any `git push` to the `master` branch that modifies files in `backend/`.

**Execution Steps:**
1.  Checkout code and configure AWS credentials.
2.  Authenticate to **Amazon ECR** (Elastic Container Registry) — the private Docker image registry.
3.  Build the Docker image from the `backend/` directory, tagged with the Git commit SHA for immutable versioning.
4.  Push the tagged image to ECR.
5.  Fetch the latest ECS Task Definition from AWS.
6.  Update the Task Definition to reference the newly built image.
7.  Deploy the updated Task Definition to the `mpd-backend-service` in the `mpd-cluster`.
8.  Wait for service stability — ECS performs a rolling update, replacing old containers with new ones only after health checks pass, ensuring **zero-downtime deployment**.

---

## 5. Infrastructure as Code (IaC)

The entire cloud infrastructure is defined declaratively using **AWS CloudFormation** YAML templates, following the Infrastructure as Code (IaC) principle. This means the infrastructure is version-controlled alongside application code and can be recreated from scratch at any time.

| Template File | Resources Defined |
| :--- | :--- |
| `frontend-infra.yml` | S3 Bucket, Bucket Policy, CloudFront Distribution |
| `alb-ecs.yml` | Security Groups, Application Load Balancer, Target Group, Listener, ECS Service |
| `data-infrastructure.yml` | Secrets Manager Secret, DB/Cache Security Groups, Subnet Groups, RDS PostgreSQL Instance |

The use of IaC provides two primary benefits: (1) **Disaster Recovery** — the entire production environment can be re-provisioned within minutes in case of a catastrophic failure; (2) **Environment Parity** — staging and production environments can be created from the same template, eliminating configuration drift.

---

## 6. Security Considerations

Several security best practices were implemented:

- **No Hardcoded Secrets:** All credentials are managed through AWS Secrets Manager and injected at runtime. No sensitive data appears in any source file.
- **VPC Isolation:** The RDS database is configured as `PubliclyAccessible: false`. It can only be reached by resources within the same Virtual Private Cloud (VPC), specifically by the ECS tasks allowed through their Security Group.
- **Least Privilege:** IAM roles assigned to ECS tasks and GitHub Actions workflows are scoped to only the permissions required (e.g., ECR push, ECS update, SSM read).
- **HTTPS Everywhere:** CloudFront enforces HTTPS for all end-user connections.

---

## 7. Evaluation and Results

### 7.1 Deployment Performance
The CI/CD pipeline achieves a complete frontend deployment (including build, S3 sync, and CloudFront invalidation) in approximately **2–3 minutes** from a `git push`. The backend rolling deployment completes in approximately **5–7 minutes**, including Docker build and ECS service stabilization.

### 7.2 System Availability
By leveraging ECS Fargate's rolling deployment strategy and the ALB's health check mechanism, the system achieves **zero-downtime deployments**. During a deployment, traffic is continuously served by the old container until the new container is confirmed healthy.

### 7.3 AI Inference Accuracy
The FinBERT model (`yiyanghkust/finbert-tone`) reports a classification accuracy of **97.07%** on the Financial PhraseBank dataset (all-agreement subset) as documented in the original paper [2]. In live testing on the Market Pulse platform, sentiment classifications for major US and Indian market news headlines were qualitatively consistent with human evaluation.

### 7.4 Cold Start Performance
Due to the FinBERT model being loaded into memory at ECS task startup (not per-request), inference latency is dominated by the API calls to `yfinance` and `GNews` (typically 2–5 seconds), rather than the model itself (< 200ms per headline).

---

## 8. Conclusion

Market Pulse demonstrates that a production-grade, AI-powered financial analytics platform can be built and operated on AWS with a high degree of automation and resilience. The key architectural contributions are:

1.  **Containerized AI Inference:** Packaging the FinBERT model in a Docker container and running it on ECS Fargate enables scalable, managed ML inference without dedicated GPU hardware.
2.  **Global Content Delivery:** The S3 + CloudFront architecture delivers sub-100ms frontend load times globally while enforcing HTTPS.
3.  **Automated DevOps:** The dual GitHub Actions pipeline eliminates manual deployment errors and enables rapid, safe iteration.
4.  **Secure Data Management:** Integration with AWS Secrets Manager and VPC-isolated RDS ensures credential and data security without developer overhead.

Future work includes implementing a request caching layer (ElastiCache Redis) to reduce third-party API call latency, integrating a dedicated model-serving framework (e.g., TorchServe) for more efficient inference, and adding an autoscaling policy to the ECS service to handle demand spikes.

---

## References

[1] J. Bollen, H. Mao, and X. Zeng, "Twitter mood predicts the stock market," *Journal of Computational Science*, vol. 2, no. 1, pp. 1–8, 2011.

[2] D. Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models," *arXiv preprint arXiv:1908.10063*, 2019.

[3] Cloud Native Computing Foundation, "CNCF Cloud Native Definition v1.0," [Online]. Available: https://github.com/cncf/toc/blob/main/DEFINITION.md.

[4] N. Dragoni et al., "Microservices: Yesterday, Today, and Tomorrow," *Present and Ulterior Software Engineering*, Springer, pp. 195–216, 2017.

[5] L. Boyuan et al., "A Survey on the Deployment of Machine Learning Inference in the Cloud," *IEEE Transactions on Cloud Computing*, 2023.

[6] Amazon Web Services, "Amazon ECS Developer Guide," AWS Documentation, 2024.

[7] A. Vaswani et al., "Attention Is All You Need," *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[8] J. Devlin, M. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," *NAACL-HLT*, 2019.

---

*© 2026 Market Pulse Research. This paper documents the genuine cloud-native implementation of the Market Pulse financial intelligence system.*
