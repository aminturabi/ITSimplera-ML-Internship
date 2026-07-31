import nbformat as nbf
import os

def create_week5_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # Title & Introduction
    cells.append(nbf.v4.new_markdown_cell("""# Week 5 Internship Task: Natural Language Processing & Sentiment Analysis

**Objective**: Build an end-to-end NLP sentiment classification pipeline on real Amazon product customer reviews. Clean raw text, convert to TF-IDF numerical vectors, train multiple text classification models, evaluate performance, and save the best performing model artifact.

---
## Part 1 Pipeline Roadmap
1. Data Loading & Exploratory Data Analysis (EDA)
2. Text Cleaning & Preprocessing (Noise reduction, Stopword removal, Lemmatization)
3. Exploratory Text Visualizations (Positive & Negative Word Clouds)
4. Feature Extraction using TF-IDF Vectorization
5. Train-Test Data Splitting
6. Model Training (Logistic Regression, Naive Bayes, Random Forest)
7. Model Evaluation & Confusion Matrix Visualizations
8. Model & Vectorizer Export for Streamlit Dashboard
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import sys
import os
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

# Add parent directory to path to import modular cleaner
sys.path.append(os.path.abspath('..'))
from utils.nlp_cleaner import clean_text

# Style configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Segoe UI'
plt.rcParams['font.size'] = 11
print("Libraries imported successfully!")
"""))

    # 1. Load Dataset
    cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Loading & Inspection

We load the **Amazon Product Reviews Dataset** (`DataSet (W5).csv`) and inspect its structural properties, columns, and feedback sentiment distribution.
"""))

    cells.append(nbf.v4.new_code_cell("""data_path = os.path.join('..', 'data', 'DataSet (W5).csv')
if not os.path.exists(data_path):
    data_path = os.path.join('data', 'DataSet (W5).csv')

df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)
print("\\nFirst 5 Rows:")
display(df.head())

print("\\nDataset Info:")
df.info()

print("\\nMissing Values:")
print(df.isnull().sum())
"""))

    cells.append(nbf.v4.new_code_cell("""# Inspect class distribution of 'feedback' label (1 = Positive, 0 = Negative)
feedback_counts = df['feedback'].value_counts()
feedback_pct = df['feedback'].value_counts(normalize=True) * 100

dist_df = pd.DataFrame({'Count': feedback_counts, 'Percentage (%)': feedback_pct})
dist_df.index = ['Positive (1)', 'Negative (0)']
display(dist_df)

# Plot class distribution
plt.figure(figsize=(6, 4))
ax = sns.barplot(x=dist_df.index, y=dist_df['Count'], palette=['#2ecc71', '#e74c3c'])
plt.title('Sentiment Class Distribution in Amazon Reviews', fontsize=14, fontweight='bold', pad=12)
plt.ylabel('Number of Reviews')
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    # 2. Text Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""## 2. Text Cleaning & Preprocessing

Raw text in customer reviews contains noise such as HTML tags, URLs, special characters, numbers, and common English stopwords ('the', 'is', 'and') that carry minimal sentiment signal.

We apply a comprehensive cleaning pipeline using our modular function `clean_text()`:
- Lowercasing
- Removal of HTML tags, URLs, numbers, and special characters
- Tokenization & stopword removal
- Wordnet Lemmatization (reducing words to canonical base forms)
"""))

    cells.append(nbf.v4.new_code_cell("""# Apply cleaning to verified_reviews
df['cleaned_reviews'] = df['verified_reviews'].astype(str).apply(clean_text)

# Inspect before and after cleaning samples
sample_comparison = df[['verified_reviews', 'cleaned_reviews', 'feedback']].head(5)
for idx, row in sample_comparison.iterrows():
    print(f"--- Sample {idx+1} [Sentiment: {row['feedback']}] ---")
    print(f"RAW    : {row['verified_reviews']}")
    print(f"CLEANED: {row['cleaned_reviews']}\\n")
"""))

    # 3. Word Cloud Generation
    cells.append(nbf.v4.new_markdown_cell("""## 3. Word Cloud Visualizations

We generate two separate Word Clouds to contrast the most frequent and salient terms present in **Positive Reviews** versus **Negative Reviews**.
"""))

    cells.append(nbf.v4.new_code_cell("""pos_text = " ".join(df[df['feedback'] == 1]['cleaned_reviews'])
neg_text = " ".join(df[df['feedback'] == 0]['cleaned_reviews'])

wc_pos = WordCloud(width=800, height=400, background_color='#111827', colormap='Greens').generate(pos_text)
wc_neg = WordCloud(width=800, height=400, background_color='#111827', colormap='Reds').generate(neg_text)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

axes[0].imshow(wc_pos, interpolation='bilinear')
axes[0].set_title('Word Cloud - Positive Reviews (Feedback = 1)', fontsize=16, fontweight='bold', color='#10B981', pad=15)
axes[0].axis('off')

axes[1].imshow(wc_neg, interpolation='bilinear')
axes[1].set_title('Word Cloud - Negative Reviews (Feedback = 0)', fontsize=16, fontweight='bold', color='#EF4444', pad=15)
axes[1].axis('off')

plt.tight_layout()
plt.show()
"""))

    # 4. Text Vectorization (TF-IDF Explanation)
    cells.append(nbf.v4.new_markdown_cell("""## 4. Text Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency)

### Why TF-IDF was chosen:
1. **Downweighting Frequent Terms**: Unlike simple Bag-of-Words (CountVectorizer) which counts raw word frequencies, **TF-IDF** penalizes words that appear frequently across *all* documents (e.g., product names like "echo", "alexa" or general words like "got").
2. **Highlighting Discriminative Words**: Words that appear frequently in a *specific* review but rarely across the entire corpus receive higher TF-IDF weights, directly sharpening the classifier's ability to pick up strong sentiment indicators (e.g., "defective", "amazing", "refused").
3. **N-Gram Support**: We include unigrams and bigrams (`ngram_range=(1, 2)`), capturing key two-word sentiment patterns like "not working" or "great product".
"""))

    cells.append(nbf.v4.new_code_cell("""vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned_reviews'])
y = df['feedback']

print(f"TF-IDF Feature Matrix Shape: {X.shape}")
print("Sample extracted features (first 20 vocabulary items):")
print(vectorizer.get_feature_names_out()[:20])
"""))

    # 5. Train-Test Split
    cells.append(nbf.v4.new_markdown_cell("""## 5. Train-Test Dataset Splitting

We perform an 80/20 train-test split using **stratified sampling** to maintain identical class ratios in both training and test sets.
"""))

    cells.append(nbf.v4.new_code_cell("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size : {X_test.shape[0]} samples")
"""))

    # 6. Model Training & Evaluation
    cells.append(nbf.v4.new_markdown_cell("""## 6. Model Training & Evaluation

We train and compare three distinct classification algorithms:
1. **Logistic Regression**: Linear baseline well-suited for high-dimensional text vectors.
2. **Multinomial Naive Bayes**: Probabilistic model suited for word frequency distributions.
3. **Random Forest Classifier**: Non-linear ensemble model capturing non-linear feature interactions.
"""))

    cells.append(nbf.v4.new_code_cell("""models = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'Multinomial Naive Bayes': MultinomialNB(alpha=0.5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
confusion_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    }
    confusion_matrices[name] = cm

    print(f"=== {name} Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))
"""))

    # Confusion Matrix Visualization
    cells.append(nbf.v4.new_markdown_cell("""### Confusion Matrix Visualizations"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, cm) in zip(axes, confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Pred Neg', 'Pred Pos'],
                yticklabels=['Actual Neg', 'Actual Pos'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f'{name}\\nConfusion Matrix', fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
plt.show()
"""))

    # Model Performance Chart & Selection
    cells.append(nbf.v4.new_markdown_cell("""## 7. Performance Comparison & Model Persistence

We compare metrics across all models and save the best performing model along with the TF-IDF vectorizer.
"""))

    cells.append(nbf.v4.new_code_cell("""results_df = pd.DataFrame(results).T
display(results_df.style.highlight_max(axis=0, color='lightgreen'))

# Plot performance chart
plt.figure(figsize=(10, 5))
ax = results_df.plot(kind='bar', figsize=(11, 6), colormap='viridis')
plt.title('Classifier Performance Comparison', fontsize=14, fontweight='bold', pad=12)
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.xticks(rotation=0, fontweight='bold')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Determine best model based on F1-Score
best_model_name = results_df['F1-Score'].idxmax()
best_f1 = results_df.loc[best_model_name, 'F1-Score']
best_model_obj = models[best_model_name]

print(f"Top Performing Model: {best_model_name} with F1-Score of {best_f1:.4f}")

# Export best model and TF-IDF vectorizer
model_dir = os.path.join('..', 'models')
if not os.path.exists(model_dir):
    model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'best_sentiment_model.pkl')
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

joblib.dump(best_model_obj, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"Saved Best Model to: {os.path.abspath(model_path)}")
print(f"Saved Vectorizer to: {os.path.abspath(vectorizer_path)}")
"""))

    nb['cells'] = cells
    nb['metadata'] = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.11.0'
        }
    }

    os.makedirs('notebooks', exist_ok=True)
    nb_path = os.path.join('notebooks', 'week5_nlp.ipynb')
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created notebook structure at {nb_path}")

if __name__ == "__main__":
    create_week5_notebook()
