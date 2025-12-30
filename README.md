📘 Narrative Biomarkers for Depression Detection
A Machine Learning & NLP Project Using the DAIC-WOZ Dataset (MSc Data Science Dissertation)

This repository contains my MSc Data Science final project, which investigates how linguistic patterns in patient narratives can be used to detect depression risk.
The project combines machine learning, BERT embeddings, explainable AI (XAI), and fairness auditing, and includes an interactive narrative analysis dashboard.

⭐ Project Summary

Traditional depression screening requires expert clinicians and structured interviews. This project explores how Natural Language Processing (NLP) can support early detection by analysing how people talk about themselves, their emotions, and their experiences.

Using the DAIC-WOZ clinical interview dataset, I built an explainable machine learning system capable of predicting depression risk from text while highlighting important linguistic features and ensuring fairness.

🔍 Key Features
✔️ 1. Machine Learning Pipeline

DAIC-WOZ transcript preprocessing

Feature extraction:

TF-IDF

LIWC-style linguistic features

BERT embeddings

Models implemented:

Logistic Regression

Random Forest

BERT + Dense Neural Network (best model)

✔️ 2. Model Performance

Best F1-Score: 0.78

ROC-AUC: 0.85

Evaluated using stratified cross-validation

Includes confusion matrix and feature importance charts

✔️ 3. Explainable AI (XAI)

SHAP global feature importance

LIME-style local narrative highlighting

Identify narrative biomarkers such as:

Self-focus language (e.g., I, me, my)

Negative emotion terms

Cognitive distortions

✔️ 4. Fairness Audit

Subgroup performance analysis (gender & age)

Reduction in false-negative disparity after mitigation

Ethical framework for responsible use

✔️ 5. Narrative Risk Dashboard

A lightweight HTML/JS dashboard that visualises:

Feature importance

Model comparison

Fairness metrics

Local narrative explanations

📄 File: narrative dashboard.html
🖼 Ready to open in any browser.

🧪 Repository Structure
/
├── Depression_AVEC2017.ipynb       # Main modelling notebook
├── narrative dashboard.html        # Interactive dashboard
├── docs/
│     └── Dissertation.pdf          # Full MSc dissertation (optional)
├── figures/                        # SHAP charts & evaluation plots
├── README.md
└── requirements.txt

🚀 Getting Started
1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>
cd <repo-name>

2. Install dependencies
pip install -r requirements.txt

3. Run the modelling notebook

Open:

Depression_AVEC2017.ipynb

4. Launch the dashboard

Open:

narrative dashboard.html


in any web browser.

⚖️ Ethics & Limitations

This project is not a clinical tool.
It is intended for academic research only.

Limitations include:

Only DAIC-WOZ dataset used

Dashboard uses simulated SHAP/LIME outputs for performance

BERT model may not generalise to all populations

Risk of misinterpretation without human oversight

A full ethical discussion can be found in the dissertation.

🧠 Skills Demonstrated

This project demonstrates:

NLP (TF-IDF, LIWC, BERT)

Machine learning (Logistic Regression, Random Forest)

Deep learning (transformers)

Explainable AI (SHAP, LIME)

Model fairness & bias mitigation

Dashboard development (HTML, Tailwind, JS)

Academic research & reporting

📚 Dissertation

You can find the full write-up in:
📄 24071456 -Project final.docx

✨ Acknowledgements

This project was completed as part of the MSc Data Science programme at UWE Bristol.
