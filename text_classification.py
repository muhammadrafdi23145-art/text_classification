import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings

warnings.filterwarnings('ignore')

def run_pipeline(csv_path):
    # 1. Load Dataset
    data = pd.read_csv(csv_path)
    data.dropna(subset=['clean_text', 'is_depression'], inplace=True)
    data['is_depression'] = data['is_depression'].astype(int)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        data['clean_text'], data['is_depression'], 
        test_size=0.2, stratify=data['is_depression'], random_state=42
    )

    # 3. Feature Extraction
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. Model Training & Evaluation
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Naive Bayes": MultinomialNB()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"\n=== {name} ===\nAccuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

    # 5. Visualization
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
    plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.savefig('accuracy_comparison.png') # Simpan untuk GitHub
    plt.show()

    # 6. Save Best Model
    best_model_name = max(results, key=results.get)
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(models[best_model_name], "depression_detection_model.pkl")
    print(f"\nBest model '{best_model_name}' saved successfully!")

if __name__ == "__main__":
    # Ganti dengan nama file dataset Anda
    run_pipeline("depression_dataset_reddit_cleaned.csv")
