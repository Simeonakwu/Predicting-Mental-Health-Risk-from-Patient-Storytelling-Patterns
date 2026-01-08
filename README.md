Predicting Mental Health Risk from Patient Storytelling Patterns
📌 Project Overview

This project explores how language patterns in patient storytelling can be used to predict mental health risk, particularly depression.
Instead of relying on questionnaires or self-reports, the project analyses how people speak and express themselves to identify early warning signs.

The work was completed as part of an MSc Data Science project at the University of the West of England (UWE Bristol).

🎯 Aim of the Project

To build a transparent, ethical, and accurate machine learning system that can:

Predict mental health risk from narrative text

Identify key narrative biomarkers (linguistic signals of distress)

Provide explainable and fair predictions suitable for healthcare decision support

🧠 What the Project Does

Uses clinical interview transcripts from the DAIC-WOZ dataset

Extracts linguistic features from patient narratives

Compares:

Traditional NLP methods (LIWC, TF-IDF)

Deep learning methods (BERT embeddings)

Evaluates model performance using standard metrics

Applies Explainable AI (XAI) to explain predictions

Conducts a fairness audit to detect and reduce bias

📊 Key Results

The BERT-based model outperformed traditional models

Achieved:

F1-Score: 0.78

ROC-AUC: 0.85

Explainability analysis (SHAP/LIME) showed that:

High use of first-person pronouns

Negative emotion words
are strong indicators of mental health risk

Fairness audit reduced gender bias in false negatives by 72%

🖥️ Narrative Risk Dashboard

The project includes a prototype dashboard that:

Simulates mental health risk prediction

Highlights words in the text that increase or reduce risk

Demonstrates how AI decisions can be interpreted by humans

⚠️ Note:
The dashboard is a proof-of-concept and does not perform real-time clinical diagnosis.

🛠️ Tools & Technologies

Python

Pandas, NumPy, Scikit-learn

Transformers (BERT)

Explainable AI (SHAP, LIME – simulated)

HTML, CSS, JavaScript (Dashboard)

📁 Repository Structure

Depression_AVEC.ipynb – Model training, evaluation, and analysis

Report.docx / PDF – Full MSc dissertation

dashboard.html – Narrative Risk Dashboard prototype

README.md – Project summary

⚠️ Ethical Disclaimer

This project:

Is not a clinical diagnostic tool

Is intended for research and educational purposes only

Should only be used as a decision-support system, not as a replacement for mental health professionals

👤 Author

Akwu Simeon Ojonugwa
MSc Data Science, UWE Bristol

GitHub: https://github.com/Simeonakwu
