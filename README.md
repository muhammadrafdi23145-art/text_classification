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
1. `text_classification.py`: Main script.
2. `requirements.txt`: List of dependencies.
3. `depression_detection_model.pkl`: The trained best-performing model.
4. `tfidf_vectorizer.pkl`: The fitted TF-IDF vectorizer.

## Result Model Accuracy Comparison
<img width="889" height="590" alt="image" src="https://github.com/user-attachments/assets/968be2cc-dd91-4f59-a228-71155405d616" />

## Result Confusion Matrix
<img width="1543" height="719" alt="image" src="https://github.com/user-attachments/assets/1c47cf1c-c07d-4f4f-982b-3899f7deafdd" />

## Evaluation
=== Logistic Regression ===

Accuracy: 0.9567
              precision    recall  f1-score   support

           0       0.94      0.97      0.96       780
           1       0.97      0.94      0.96       767

    accuracy                           0.96      1547
    macro avg       0.96      0.96      0.96      1547
    weighted avg       0.96      0.96      0.96      1547


=== Random Forest ===

Accuracy: 0.9606
              precision    recall  f1-score   support

           0       0.93      0.99      0.96       780
           1       0.99      0.93      0.96       767

    accuracy                           0.96      1547
    macro avg       0.96      0.96      0.96      1547
    weighted avg       0.96      0.96      0.96      1547


=== Naive Bayes ===

Accuracy: 0.9005
              precision    recall  f1-score   support

           0       0.97      0.83      0.89       780
           1       0.85      0.97      0.91       767

    accuracy                           0.90      1547
    macro avg       0.91      0.90      0.90      1547
    weighted avg       0.91      0.90      0.90      1547
