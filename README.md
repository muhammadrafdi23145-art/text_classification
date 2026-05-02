# Depression Detection via NLP

This repository contains an end-to-end Machine Learning pipeline to detect signs of depression in social media text using Natural Language Processing (NLP).

## Key Features
- **Data Preprocessing:** Cleaning and TF-IDF Vectorization.
- **Multi-Model Comparison:** Evaluates Logistic Regression, Random Forest, and Naive Bayes.
- **Visualization:** Accuracy comparison plots and performance metrics.
- **Deployment Ready:** Exports trained models as `.pkl` files.

## Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/yourusername/depression-detection.git](https://github.com/yourusername/depression-detection.git)

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Results
The models achieve high accuracy in classifying text, with the best model being exported for production use. You can view the accuracy_comparison.png for a visual breakdown.

## File Structure
1. text_classification.py: Main script.
2. requirements.txt: List of dependencies.
3. depression_detection_model.pkl: The trained best-performing model.
4. tfidf_vectorizer.pkl: The fitted TF-IDF vectorizer.

## requirements.txt
```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
