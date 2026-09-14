# 🧠 ML Model Training & Predictive Analytics Service

[![Production Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge&logo=fastapi)](https://github.com/zeeshan-ux-AI/ml-model-training-service)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Deploy with Vercel](https://img.shields.io/badge/Deploy-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com/new)
[![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

An end-to-end Machine Learning model training pipeline, real-time prediction API, and interactive web dashboard. Designed for automated dataset ingestion, feature engineering, model evaluation, and serverless deployment on Vercel or cloud environments.

---

## 🌟 Key Features

- **Automated ML Pipeline**: Data preprocessing, feature scaling, model training, and hyperparameter tuning.
- **FastAPI Serverless Backend**: High-performance RESTful API endpoints for real-time model inference and batch prediction.
- **Interactive Analytics Dashboard**: Modern Glassmorphism UI built for real-time visualization of model precision, recall, F1-score, and loss metrics.
- **Vercel Native Deployment**: Serverless function integration configured for zero-downtime deployment.
- **Model Persistence & Artifact Management**: Automated evaluation logging and model serialization.

---

## 🏗️ Architecture Overview

```
 ┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
 │ Data Ingestion  │ ────> │  ML Training Engine  │ ────> │  FastAPI REST Endpoint │
 └─────────────────┘       └──────────────────────┘       └────────────────────────┘
                                                                       │
                                                                       ▼
                                                          ┌────────────────────────┐
                                                          │  Vercel Web UI Dashboard│
                                                          └────────────────────────┘
```

---

## 📁 Repository Structure

```
ml-model-training-service/
├── api/
│   └── index.py            # Serverless FastAPI application entry point
├── models/
│   ├── trainer.py          # Machine learning model training logic
│   └── predictor.py        # Inference pipeline & payload validation
├── public/
│   └── index.html          # Web dashboard interface for Vercel
├── requirements.txt        # Production Python dependencies
├── vercel.json             # Vercel serverless build configuration
└── README.md               # Project documentation
```

---

## 🚀 Quick Start & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/zeeshan-ux-AI/ml-model-training-service.git
cd ml-model-training-service
```

### 2. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run ML model training
```bash
python3 models/trainer.py
```

### 4. Start API Server
```bash
uvicorn api.index:app --reload --port 8000
```
Open `http://localhost:8000/docs` to view the interactive OpenAPI Swagger documentation.

---

## 🌐 Deploy to Vercel

Click the button below to deploy this ML service directly to Vercel:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fzeeshan-ux-AI%2Fml-model-training-service)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
