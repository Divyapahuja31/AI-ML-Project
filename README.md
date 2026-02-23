# Intelligent Credit Risk Scoring & Agentic Lending Decision Support System 🏦

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Enabled-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![AI/ML](https://img.shields.io/badge/AI%2FML-Fintech-brightgreen.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agents-purple.svg)

An end-to-end AI-powered fintech application that predicts borrower credit risk using an optimized Machine Learning pipeline and extends into an **Agentic AI Assistant** for autonomous, reasoning-based lending decision support.

---

## 1. 📖 Project Overview

### The Real-World Lending Problem
Traditional credit underwriting relies heavily on static scorecards and rigid, rule-based systems. These legacy systems often fail to adapt to complex financial behaviors, lack transparency into *why* an applicant was rejected, and require significant manual review time from credit analysts.

### Purpose of the System
This platform solves the manual bottleneck in modern fintech by introducing a two-phase architecture:
1. **Quantitative Risk Scoring**: Rapidly predicting the statistical likelihood of borrower default.
2. **Qualitative AI Reasoning**: Deploying an LLM-driven Agent to synthesize the data, explain the risk factors, and draft professional lending approval/rejection recommendations.

### Business & AI Motivation
By combining predictive Machine Learning (Random Forests) with generative Agentic AI (LangGraph/LLMs), lending institutions can scale loan origination, drastically reduce human underwriting time, and maintain strict, explainable compliance standards.

---

## 2. ✨ Key Features

- **ML-Based Credit Risk Prediction**: Real-time evaluation of borrower delinquency probabilities.
- **Risk Scoring**: Statistically modeling "High Risk" vs "Low Risk" applicants.
- **Explainable AI Insights**: Automated, transparent breakdowns of the strongest features driving the prediction (e.g., Credit Utilization, Debt-to-Income).
- **Interactive Streamlit UI**: A clean, modern dashboard allowing loan officers to input profile details and receive instant feedback.
- **Agentic AI Lending Assistant *(Milestone 2)***: An autonomous LLM-based agent capable of drafting formalized lending memos and querying policy documents.
- **Structured Lending Recommendations**: Natural language summaries justifying the AI’s final decision.

---

## 3. 🏗️ System Architecture

### High-Level Workflow
1. **Data Ingestion**: A borrower's financial profile is submitted via the UI.
2. **Preprocessing**: The data flows through a strict pipeline handling missing values (median imputation) and categorical text (One-Hot Encoding).
3. **ML Inference**: The processed data is fed into a persistently loaded `RandomForestClassifier`.
4. **Agentic Synthesis**: The output probabilities and feature importances are passed to the Agentic Assistant, which cross-references internal policies to generate a final recommendation.

---

## 4. 🧰 Tech Stack

- **Core Python**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, Joblib
- **Frontend / UI**: Streamlit
- **Agentic AI & LLMs**: LangGraph, Open-Source LLMs
- **Vector Databases**: FAISS / Chroma (For policy retrieval)
- **Deployment**: Streamlit Cloud / HuggingFace Spaces

---

## 5. 📂 Project Structure

```text
AI-ML-Project/
├── app/
│   └── streamlit_app.py       # Streamlit web application dashboard
├── data/                      
│   └── Credit Risk Benchmark Dataset.csv
├── models/                    
│   └── risk_model.pkl         # Trained Random Forest persistence
├── notebooks/                 
│   └── eda.ipynb              # Jupyter Notebook for Exploratory Data Analysis
├── src/
│   ├── preprocess.py          # Data cleaning, encoding, and train/test splitting pipeline
│   ├── train_model.py         # Model training script
│   └── explain_model.py       # Explainable AI feature importance extraction
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 6. ⚙️ Machine Learning Pipeline

1. **Data Preprocessing**: Handling Missing Data (NaNs) via automated median imputation to preserve sample size without skewing distribution.
2. **Feature Engineering**: One-hot encoding text values and scaling numeric fields to uniform baseline distributions.
3. **Model Training**: Utilizing a `RandomForestClassifier` initialized with `class_weight="balanced"` to strictly handle class imbalances inherent in default prediction datasets.
4. **Evaluation**: Validation utilizing Train/Test stratified splitting.
5. **Model Persistence**: Serializing the object mathematically via `joblib` into binary formats optimized for microservice inference.

---

## 7. 📈 Model Evaluation Metrics

In credit risk, accuracy alone is incredibly misleading. We optimize across multiple axes:

- **Accuracy's Flaw**: If 95% of loans perform, a model that simply predicts "Approved" for everyone achieves 95% accuracy but bankrupts the bank.
- **Precision**: Of all applicants the model flagged as "High Risk", how many actually defaulted? Crucial to avoid incorrectly declining great customers.
- **Recall**: Out of all *actual* defaults, how many did the model successfully catch? In lending, False Positives (missing a default) are immensely expensive, making Recall a top-priority metric.
- **ROC-AUC**: Evaluates the model’s overall ability to separate the "Good" borrowers from the "Bad" borrowers across all probability threshold levels.

---

## 8. 🔍 Explainable AI (XAI)

Black-box models are restricted in heavily regulated financial environments. This system guarantees explainability by extracting node-level split importances directly from the Random Forest.

Common driving risk factors identified by the system include:
1. **Revolving Utilization**: (Credit Card Balances / Total Limit)
2. **Debt Ratio**: (Total Monthly Debt Payments / Gross Income)
3. **Historical Delinquency**: Specifically the frequency of hitting 90+ days past due.

---

## 9. 🧠 Agentic AI Decision System *(Milestone 2)*

While the ML model outputs a raw probability (e.g., *78% chance of default*), an underwriter needs more context. 

The **Agentic AI System** wraps the ML model. The Agent evaluates the features autonomously, searches through internal bank policies utilizing **Retrieval-Augmented Generation (RAG)**, and issues a coherent lending memo defining exactly *why* a loan should or should not be originated based on combining strict mathematical risk with dynamic policy guidelines.

---

## 10. 🖥️ User Interface

The **Streamlit** dashboard empowers non-technical users (Loan Officers, Underwriters) to securely submit an applicant's financial attributes. 

The application instantly returns:
- A prominent High/Low-Risk Boolean alert.
- The precise statistical probability of default.
- An interactive drop-down translating the mathematical feature-importances into plain insight text.

---

## 11. 🚀 Installation & Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/your-username/AI-ML-Project.git
cd AI-ML-Project

# 2. Create the virtual environment
python3 -m venv venv

# 3. Activate the environment (Mac/Linux)
source venv/bin/activate
# Windows: venv\Scripts\activate

# 4. Install requirements
pip install -r requirements.txt

# 5. Run the Model Training Pipeline (Will generate the .pkl file)
python3 src/train_model.py

# 6. Run the local Streamlit Application
streamlit run app/streamlit_app.py
```

---

## 12. 💡 Usage Guide

1. Launch the Streamlit application.
2. In the user interface, fill out the **Applicant Details** configuration form. Adjust critical variables like `Age`, `Monthly Income`, and `Credit Utilization`. 
3. Click the **Predict Risk** button.
4. The system will process the dataset precisely as it was processed in training, generating live inferences immediately. Check the accordion menu to see the decision rationale.

---

## 13. 🌐 Deployment

This application is fully responsive and container-ready. 
For public portfolio showcasing, the dashboard is designed to be easily hosted using **Streamlit Community Cloud** or **HuggingFace Spaces** with zero backend infrastructure configuration required.

---

## 14. 🔮 Future Enhancements

- **RAG Integration**: Ingesting PDF credit policy manuals to allow the Agent to cite actual internal policy rules during application rejections.
- **Regulatory Retrieval**: Tying the LLM to live searches of local lending limits and regulations.
- **PDF Report Generation**: Allowing underwriters to export the Agent's final memo securely as a standardized PDF document.
- **Portfolio Analytics**: Building a secondary dashboard page to analyze model drift and macro-level applicant trends.

---