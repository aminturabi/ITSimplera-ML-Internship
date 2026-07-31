import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

import sys
sys.path.append(os.path.abspath('.'))
from utils.nlp_cleaner import clean_text

def run_pipeline():
    print("--- 1. Setting up directories and copying dataset ---")
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)

    source_csv = 'DataSet (W5).csv'
    target_csv = os.path.join('data', 'DataSet (W5).csv')
    if os.path.exists(source_csv) and not os.path.exists(target_csv):
        shutil.copy(source_csv, target_csv)
        print(f"Copied {source_csv} -> {target_csv}")
    elif os.path.exists(target_csv):
        print(f"Dataset already present at {target_csv}")

    print("--- 2. Loading and Inspecting Dataset ---")
    df = pd.read_csv(target_csv)
    print(f"Dataset shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nClass distribution ('feedback'):")
    print(df['feedback'].value_counts(normalize=True))
    print(df['feedback'].value_counts())

    print("--- 3. Preprocessing and Cleaning Text ---")
    df['cleaned_reviews'] = df['verified_reviews'].astype(str).apply(clean_text)
    
    # Save cleaned dataframe for app usage if needed
    cleaned_csv_path = os.path.join('data', 'cleaned_amazon_reviews.csv')
    df.to_csv(cleaned_csv_path, index=False)
    print(f"Saved cleaned dataset to {cleaned_csv_path}")

    print("--- 4. Generating Word Clouds ---")
    pos_text = " ".join(df[df['feedback'] == 1]['cleaned_reviews'])
    neg_text = " ".join(df[df['feedback'] == 0]['cleaned_reviews'])

    wordcloud_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(pos_text)
    wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)

    pos_wc_path = os.path.join('data', 'positive_wordcloud.png')
    neg_wc_path = os.path.join('data', 'negative_wordcloud.png')
    wordcloud_pos.to_file(pos_wc_path)
    wordcloud_neg.to_file(neg_wc_path)
    print(f"Saved Word Clouds: {pos_wc_path}, {neg_wc_path}")

    print("--- 5. Vectorization (TF-IDF) ---")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned_reviews'])
    y = df['feedback']

    print(f"TF-IDF feature matrix shape: {X.shape}")

    print("--- 6. Train/Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    print("--- 7. Training & Evaluating Models ---")
    models = {
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model = None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc,
            'confusion_matrix': cm,
            'model_obj': model
        }

        print(f"{name} -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f"\n>>> Best Model: {best_model_name} (F1 Score: {best_f1:.4f})")

    # Save best model and vectorizer
    model_path = os.path.join('models', 'best_sentiment_model.pkl')
    vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved best model to {model_path}")
    print(f"Saved vectorizer to {vectorizer_path}")

    # Plot model comparison chart
    plt.figure(figsize=(10, 5))
    metrics_df = pd.DataFrame(results).T[['accuracy', 'precision', 'recall', 'f1_score']]
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'model_comparison.png'))
    plt.close()
    print("Saved model comparison plot to data/model_comparison.png")

    return results, df

if __name__ == "__main__":
    run_pipeline()
