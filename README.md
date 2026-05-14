# Bank Customer Churn Prediction & Explainable AI

> Predicting bank customer churn using ensemble ML models, with SHAP and LIME to interpret model decisions and derive actionable business insights.

**Course:** 2400-DS2ML2 – Machine Learning 2  
**Authors:** Ronald Mjonono (473522) · Linxiao Mu (474569)  
**Dataset:** [Churn Modelling – Kaggle](https://www.kaggle.com/shrutimechlearn/churn-modelling)

---

## Project Overview

A European bank wants to identify customers likely to leave before they churn. This project builds a binary classification pipeline on 10,000 customers and applies Explainable AI (XAI) techniques to translate black-box model predictions into clear, actionable retention strategies.

**Key question:** *Which customers are at risk of churning — and why?*

---

## Repository Structure

```
bank-churn-xai/
├── data/
│   └── Churn_Modelling.csv        # Source dataset (10,000 customers)
├── notebooks/
│   └── churn_full_xai.ipynb       # Full modelling & XAI pipeline
├── README.md
└── .gitignore
```

---

## Models

| Model | ROC AUC | F1 Score |
|---|---|---|
| Decision Tree | 0.852 | 0.545 |
| Random Forest | 0.879 | 0.596 |
| **Gradient Boosting** ⭐ | **0.882** | **0.614** |
| Neural Network (MLP) | 0.873 | 0.612 |

All models were tuned with `GridSearchCV` (5-fold stratified cross-validation, scoring on ROC AUC).

> **Note on metrics:** The dataset has a class imbalance (80% stayed / 20% churned). Accuracy alone is misleading — ROC AUC and F1 are the primary evaluation metrics.

---

## XAI Methods

### SHAP (SHapley Additive exPlanations)
- **TreeExplainer** applied to Random Forest and Gradient Boosting
- **Global analysis:** Beeswarm plot and mean |SHAP| bar chart reveal the top drivers of churn across all customers
- **Local analysis:** Waterfall plots explain individual predictions — why a specific customer churned or stayed

### LIME (Local Interpretable Model-agnostic Explanations)
- Applied to the Neural Network (MLP), which is not supported by TreeExplainer
- Fits a local linear approximation around a single prediction
- Confirms SHAP findings across a different model and method

---

## Key Findings

1. **Age** is the strongest churn driver — risk rises sharply after 40, peaks around 50–60
2. **NumOfProducts** — customers with 3–4 products churn more (counterintuitive: more ≠ loyal)
3. **IsActiveMember** — inactive members are significantly more likely to churn
4. **Geography_Germany** — German customers churn disproportionately vs France/Spain
5. SHAP and LIME findings are **consistent across RF, GBM, and MLP** — high confidence in results

---

## Business Recommendations

| Priority | Action |
|---|---|
| 🔴 High | Target inactive customers aged 40–60 with personalised retention outreach |
| 🟡 Medium | Launch reactivation campaigns for inactive members |
| 🔵 Medium | Review product bundling — 3–4 products correlates with dissatisfaction |
| 🟣 Low | Develop Germany-specific retention strategy |

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/<your-username>/bank-churn-xai.git
cd bank-churn-xai
```

2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn shap lime jupytext
```

3. Open the notebook
```bash
jupyter lab notebooks/churn_full_xai.ipynb
```

4. Run all cells top to bottom (~15–20 minutes due to GridSearchCV)

---

## Requirements

- Python 3.10+
- numpy < 2.0 (required for SHAP/sklearn compatibility)
- scikit-learn
- shap
- lime
- pandas, matplotlib, seaborn

---

## Limitations

- Dataset sourced from Kaggle for educational purposes — may not reflect real bank behaviour
- Churn reasons are unobserved — the model captures correlations, not causal mechanisms
- Class imbalance (80/20) was not addressed with resampling (e.g. SMOTE)
- LIME explanations are local approximations and may vary across random seeds