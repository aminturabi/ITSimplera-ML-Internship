import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from wordcloud import WordCloud

# Import modular text cleaner
from utils.nlp_cleaner import clean_text

# Page Configuration
st.set_page_config(
    page_title="Amazon Review Sentiment Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1E2F 0%, #2A2A40 100%);
        padding: 2rem;
        border-radius: 16px;
        color: #FFFFFF;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        color: #FF9900 !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: #1A1C23;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #2D3748;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FF9900;
    }
    
    .metric-label {
        color: #A0AEC0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #FF9900 0%, #FFB84D 100%);
        color: #111;
        font-weight: 700;
        border: none;
        padding: 0.6rem 1.8rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255,153,0,0.4);
    }
    
    .pos-badge {
        background-color: #10B981;
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    .neg-badge {
        background-color: #EF4444;
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load models and data
@st.cache_resource
def load_artifacts():
    model_path = os.path.join('models', 'best_sentiment_model.pkl')
    vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Saved model or vectorizer not found! Please run training script first.")
        return None, None
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

@st.cache_data
def load_dataset():
    data_path = os.path.join('data', 'DataSet (W5).csv')
    if not os.path.exists(data_path):
        data_path = 'DataSet (W5).csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df
    return None

model, vectorizer = load_artifacts()
df_raw = load_dataset()

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=160)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "📌 Navigation",
    ["Home", "Data Overview", "Sentiment Predictor"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Status")
if model is not None and vectorizer is not None:
    st.sidebar.success("Model & Vectorizer Loaded")
    st.sidebar.info(f"**Classifier**: {type(model).__name__}")
    st.sidebar.info(f"**Vocab Size**: {len(vectorizer.get_feature_names_out()):,} terms")
else:
    st.sidebar.error("Model Not Loaded")

# Page 1: Home
if page == "Home":
    st.markdown("""
    <div class="main-header">
        <h1>📦 Amazon Customer Sentiment Dashboard</h1>
        <p>An Interactive Natural Language Processing (NLP) & Sentiment Classification Application</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("🎯 Project Overview")
        st.write("""
        Natural Language Processing (NLP) allows computers to process, understand, and classify human language text. 
        This application leverages customer reviews collected from Amazon products to automate sentiment analysis, 
        evaluating whether feedback is **Positive** (satisfied customer) or **Negative** (dissatisfied customer).
        """)
        
        st.subheader("🛠️ Technical Architecture & Pipeline")
        st.markdown("""
        1. **Text Preprocessing**: Raw reviews are cleaned by removing HTML tags, URLs, numbers, special characters, and English stopwords, followed by Lemmatization using NLTK.
        2. **TF-IDF Vectorization**: Text tokens are converted into numerical feature vectors using Term Frequency-Inverse Document Frequency (5,000 max features, unigrams & bigrams).
        3. **Machine Learning Classifier**: Trained on historical review data using Logistic Regression, Naive Bayes, and Random Forest models.
        4. **Streamlit Deployment**: Real-time interactive inference dashboard.
        """)

    with col2:
        st.subheader("📊 Quick Dataset Summary")
        if df_raw is not None:
            total_reviews = len(df_raw)
            pos_reviews = (df_raw['feedback'] == 1).sum()
            neg_reviews = (df_raw['feedback'] == 0).sum()
            pos_pct = (pos_reviews / total_reviews) * 100
            
            st.metric("Total Amazon Reviews", f"{total_reviews:,}")
            st.metric("Positive Reviews (1)", f"{pos_reviews:,} ({pos_pct:.1f}%)")
            st.metric("Negative Reviews (0)", f"{neg_reviews:,} ({100-pos_pct:.1f}%)")
        else:
            st.warning("Dataset file not available.")

# Page 2: Data Overview
elif page == "Data Overview":
    st.title("📊 Data Overview & Visualizations")
    st.write("Explore the underlying Amazon product review dataset, class distribution, and word cloud patterns.")
    
    if df_raw is not None:
        col1, col2, col3 = st.columns(3)
        total_rev = len(df_raw)
        pos_rev = (df_raw['feedback'] == 1).sum()
        neg_rev = (df_raw['feedback'] == 0).sum()
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_rev:,}</div>
                <div class="metric-label">Total Reviews</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#10B981">{pos_rev:,}</div>
                <div class="metric-label">Positive (Feedback=1)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#EF4444">{neg_rev:,}</div>
                <div class="metric-label">Negative (Feedback=0)</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Donut Chart for Class Distribution
        chart_col, table_col = st.columns([1, 1])
        
        with chart_col:
            st.subheader("📈 Class Sentiment Distribution")
            fig = px.pie(
                names=['Positive (1)', 'Negative (0)'],
                values=[pos_rev, neg_rev],
                color=['Positive (1)', 'Negative (0)'],
                color_discrete_map={'Positive (1)': '#10B981', 'Negative (0)': '#EF4444'},
                hole=0.4
            )
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
            
        with table_col:
            st.subheader("📋 Dataset Preview")
            st.dataframe(df_raw[['rating', 'variation', 'verified_reviews', 'feedback']].head(10), use_container_width=True)
            
        st.markdown("---")
        st.subheader("☁️ Word Clouds (Positive vs. Negative Reviews)")
        
        pos_img_path = os.path.join('data', 'positive_wordcloud.png')
        neg_img_path = os.path.join('data', 'negative_wordcloud.png')
        
        if os.path.exists(pos_img_path) and os.path.exists(neg_img_path):
            wc_col1, wc_col2 = st.columns(2)
            with wc_col1:
                st.markdown("### 🟢 Positive Reviews Word Cloud")
                st.image(pos_img_path, use_container_width=True)
            with wc_col2:
                st.markdown("### 🔴 Negative Reviews Word Cloud")
                st.image(neg_img_path, use_container_width=True)
        else:
            # Generate on the fly if images not found
            with st.spinner("Generating Word Clouds..."):
                cleaned_series = df_raw['verified_reviews'].astype(str).apply(clean_text)
                pos_text = " ".join(cleaned_series[df_raw['feedback'] == 1])
                neg_text = " ".join(cleaned_series[df_raw['feedback'] == 0])
                
                wc_pos = WordCloud(width=800, height=400, background_color='#111827', colormap='Greens').generate(pos_text)
                wc_neg = WordCloud(width=800, height=400, background_color='#111827', colormap='Reds').generate(neg_text)
                
                wc_col1, wc_col2 = st.columns(2)
                with wc_col1:
                    st.markdown("### 🟢 Positive Reviews Word Cloud")
                    st.image(wc_pos.to_array(), use_container_width=True)
                with wc_col2:
                    st.markdown("### 🔴 Negative Reviews Word Cloud")
                    st.image(wc_neg.to_array(), use_container_width=True)
    else:
        st.warning("Dataset not found. Please check data folder.")

# Page 3: Sentiment Predictor
elif page == "Sentiment Predictor":
    st.title("🔮 Real-Time Sentiment Predictor")
    st.write("Type or paste any product review below to instantly predict its sentiment and confidence score.")
    
    st.markdown("### 💡 Quick Try Samples")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    default_text = ""
    
    if sample_col1.button("🟢 Sample Positive Review"):
        st.session_state['user_review'] = "Love my Echo! Great sound quality, crisp speaker, and easy to set up. Alexa answers instantly!"
    if sample_col2.button("🔴 Sample Negative Review"):
        st.session_state['user_review'] = "Terrible product. It stopped working after two days and customer support refused to refund."
    if sample_col3.button("🟡 Sample Mixed Review"):
        st.session_state['user_review'] = "The speaker looks nice on my desk but the setup process was confusing and it disconnects often."

    user_input = st.text_area(
        "Enter Product Review Text:",
        value=st.session_state.get('user_review', ''),
        height=140,
        placeholder="Type a review here, e.g., 'This speaker exceeded all my expectations! Sound is fantastic.'"
    )
    
    analyze_btn = st.button("🚀 Predict Sentiment")
    
    if analyze_btn or user_input.strip():
        if not user_input.strip():
            st.warning("Please type a review first!")
        elif model is None or vectorizer is None:
            st.error("Model or vectorizer is not loaded. Run training script first.")
        else:
            # 1. Apply exact same text cleaning
            cleaned_input = clean_text(user_input)
            
            # 2. Vectorize text
            vectorized_input = vectorizer.transform([cleaned_input])
            
            # 3. Predict sentiment & confidence probabilities
            prediction = model.predict(vectorized_input)[0]
            probabilities = model.predict_proba(vectorized_input)[0]
            
            prob_neg, prob_pos = probabilities[0], probabilities[1]
            confidence = prob_pos if prediction == 1 else prob_neg
            
            st.markdown("---")
            st.subheader("📊 Sentiment Analysis Results")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.markdown("#### **Predicted Sentiment:**")
                if prediction == 1:
                    st.markdown('<div class="pos-badge">🟢 POSITIVE SENTIMENT</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="neg-badge">🔴 NEGATIVE SENTIMENT</div>', unsafe_allow_html=True)
                    
                st.markdown(f"<br><strong>Model Confidence Score:</strong> {confidence*100:.2f}%", unsafe_allow_html=True)
                st.progress(float(confidence))
                
                with st.expander("🔍 View Preprocessed Clean Text"):
                    st.write("**Original Text:**", user_input)
                    st.write("**Cleaned Text (Tokens & Lemmatized):**", cleaned_input if cleaned_input else "*(No meaningful tokens after stopword removal)*")

            with res_col2:
                st.markdown("#### **Class Probabilities Breakdown**")
                prob_fig = go.Figure(go.Bar(
                    x=[prob_pos * 100, prob_neg * 100],
                    y=['Positive', 'Negative'],
                    orientation='h',
                    marker=dict(color=['#10B981', '#EF4444']),
                    text=[f"{prob_pos*100:.1f}%", f"{prob_neg*100:.1f}%"],
                    textposition='auto'
                ))
                prob_fig.update_layout(
                    xaxis=dict(range=[0, 100], title="Probability (%)"),
                    height=220,
                    margin=dict(l=20, r=20, t=30, b=30)
                )
                st.plotly_chart(prob_fig, use_container_width=True)
