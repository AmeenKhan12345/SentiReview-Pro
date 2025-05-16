import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import string
from lime.lime_text import LimeTextExplainer
import numpy as np
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer

# Page configuration
st.set_page_config(
    page_title="SentiReview Pro: AI-Powered Review Analysis",
    page_icon="✨",
    layout="wide"
)

# Custom CSS for improved appearance
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1E3A8A;
        text-align: center;
    }
    .subheader {
        font-family: 'Helvetica Neue', sans-serif;
        color: #3B82F6;
        text-align: center;
        margin-bottom: 30px;
    }
    .stProgress > div > div > div > div {
        background-color: #10B981;
    }
    .highlight-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #F3F4F6;
        margin-bottom: 20px;
    }
    .positive-badge {
        background-color: #10B981;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    .negative-badge {
        background-color: #EF4444;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    .explanation-title {
        font-size: 20px;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 14px;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='main-header'>✨ SentiReview Pro ✨</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subheader'>Advanced AI-Powered Sentiment Analysis for Product Reviews</h3>", unsafe_allow_html=True)

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📝 Single Review Analysis", "📊 Batch Analysis", "ℹ️ About"])

# Load the saved model and vectorizer
@st.cache_resource
def load_models():
    try:
        model = joblib.load("svm_sentiment_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("🚫 Model files not found. Please ensure the model files are in the correct location.")
        return None, None

model, vectorizer = load_models()

# Download NLTK resources (they install into a writable cache)
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(ENGLISH_STOP_WORDS)
lemmatizer = WordNetLemmatizer()

#Function to preprocess text
def preprocess_text(text):
    if pd.isnull(text) or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Strip whitespace
    text = text.strip()
    
    # Tokenize (whitespace split)
    tokens = text.split()
    # Remove stopwords and lemmatize
    processed_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words
    ]
    return ' '.join(processed_tokens)
    
# Initialize LIME Explainer
explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

def explain_prediction(review_text):
    def predict_proba(texts):
        processed_texts = [preprocess_text(t) for t in texts]
        vecs = vectorizer.transform(processed_texts)
        decision = model.decision_function(vecs)
        # Convert decision function to probabilities using sigmoid
        probs = np.vstack([1 / (1 + np.exp(decision)), 1 - 1 / (1 + np.exp(decision))]).T
        return probs

    exp = explainer.explain_instance(review_text, predict_proba, num_features=10)
    return exp

def highlight_lime_words(explanation):
    highlighted_text = ""
    for word, weight in explanation:
        if weight > 0:
            color = f"rgba(16, 185, 129, {min(abs(weight), 1)})"  # Green for positive
        else:
            color = f"rgba(239, 68, 68, {min(abs(weight), 1)})"  # Red for negative
        highlighted_text += f"<span style='background-color:{color}; padding:2px; margin:2px; border-radius:3px'>{word}</span> "
    return highlighted_text.strip()

# Create a function to get representative samples from your dataset
# In production, you should replace this with actual data from your training set
def get_representative_samples(n=50):
    positive_examples = [
        "The product exceeded my expectations in every way!",
        "Incredible value for money, highly recommend it.",
        "Exactly what I was looking for, amazing quality.",
        "Fast shipping and excellent customer service.",
        "Works perfectly, very happy with my purchase."
    ]
    
    negative_examples = [
        "Broke after a week, very poor quality.",
        "Terrible customer service, would not buy again.",
        "Not as described, complete waste of money.",
        "Shipping took forever and the product arrived damaged.",
        "Doesn't work as advertised, very disappointed."
    ]
    
    neutral_examples = [
        "It's okay but not great for the price.",
        "Average product, nothing special.",
        "Some good features but has a few issues.",
        "Expected better quality, but it works.",
        "Decent product with room for improvement."
    ]
    
    # Combine and ensure we have requested number of samples
    all_samples = positive_examples + negative_examples + neutral_examples
    result = all_samples * (n // len(all_samples) + 1)
    return result[:n]

# Function to generate SHAP explanation
def generate_shap_explanation(review_clean):
    if not model or not vectorizer:
        return None, None
    
    try:
        # Get more representative background samples
        background_texts = get_representative_samples(50)
        background_texts_processed = [preprocess_text(t) for t in background_texts]
        background_vecs = vectorizer.transform(background_texts_processed)
        
        # Transform the current review
        review_vec = vectorizer.transform([review_clean])
        
        # Create SHAP explainer with proper background
        explainer_shap = shap.LinearExplainer(model, background_vecs, feature_perturbation="interventional")
        shap_values = explainer_shap.shap_values(review_vec)
        feature_names = vectorizer.get_feature_names_out()
        
        # Create summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, 
            features=review_vec, 
            feature_names=feature_names, 
            show=False,
            plot_size=(10, 6)
        )
        
        # Save plot to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Get top influence words
        feature_importance = list(zip(feature_names, shap_values[0]))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = feature_importance[:10]
        
        return buf, top_features
        
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {str(e)}")
        return None, None

# Single Review Analysis Tab
with tab1:
    st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
    st.markdown("### 📝 Enter Your Product Review")
    
    # Example selector
    example_option = st.selectbox(
        "Try with an example or write your own review",
        ["-- Write your own --", 
         "I absolutely love this product! It works perfectly and the quality is excellent.",
         "This is the worst purchase I've ever made. It broke after two days and customer service was unhelpful.",
         "It's an okay product. Does the job but nothing special for the price."]
    )
    
    if example_option == "-- Write your own --":
        review = st.text_area("Type your review here:", "", height=150)
    else:
        review = st.text_area("Review text:", example_option, height=150)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")
    
    if analyze_button and review:
        with st.spinner("Analyzing your review..."):
            if model and vectorizer:
                review_clean = preprocess_text(review)
                review_vec = vectorizer.transform([review_clean])
                y_probs = model.decision_function(review_vec)
                prediction = 'Positive' if y_probs[0] > 0 else 'Negative'
                
                # Calculate confidence
                prob_norm = min(1.0, max(0.0, (y_probs[0] + 1) / 2)) if prediction == 'Positive' else min(1.0, max(0.0, 1 - (y_probs[0] + 1) / 2))
                
                # Display Result
                st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
                st.markdown("### 📊 Analysis Results")
                
                # Show sentiment badge
                if prediction == "Positive":
                    st.markdown(f"<p>Sentiment: <span class='positive-badge'>✓ POSITIVE</span> with {round(prob_norm*100, 1)}% confidence</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p>Sentiment: <span class='negative-badge'>✗ NEGATIVE</span> with {round(prob_norm*100, 1)}% confidence</p>", unsafe_allow_html=True)
                
                # Confidence meter
                st.progress(prob_norm)
                
                # LIME Explanations
                st.markdown("<p class='explanation-title'>🔍 Key Influential Words</p>", unsafe_allow_html=True)
                
                # Create columns for the visualization
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    explanation = explain_prediction(review)
                    st.markdown(highlight_lime_words(explanation.as_list()), unsafe_allow_html=True)
                
                with col2:
                    # Show top influential words as a table
                    influence_data = []
                    for word, weight in explanation.as_list():
                        sentiment = "Positive" if weight > 0 else "Negative"
                        influence = abs(weight)
                        influence_data.append({"Word": word, "Sentiment": sentiment, "Influence": influence})
                    
                    influence_df = pd.DataFrame(influence_data)
                    st.dataframe(
                        influence_df,
                        hide_index=True,
                        column_config={
                            "Word": st.column_config.TextColumn("Word"),
                            "Sentiment": st.column_config.TextColumn("Impact"),
                            "Influence": st.column_config.ProgressColumn(
                                "Strength",
                                min_value=0,
                                max_value=max(influence_df['Influence']),
                                format="%.2f"
                            )
                        }
                    )
                
                # SHAP Explanation
                st.markdown("<p class='explanation-title'>🧠 Deep Learning Explanation (SHAP)</p>", unsafe_allow_html=True)
                
                shap_plot, top_features = generate_shap_explanation(review_clean)
                
                if shap_plot and top_features:
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Display SHAP plot
                        st.image(shap_plot, caption="SHAP Feature Importance", use_container_width=True)
                    
                    with col2:
                        # Display top features in a table
                        st.markdown("**Top Influential Features**")
                        
                        feature_df = pd.DataFrame([(word, float(value)) for word, value in top_features], 
                                                columns=["Feature", "Impact"])
                        feature_df["Direction"] = feature_df["Impact"].apply(lambda x: "Positive" if x > 0 else "Negative")
                        feature_df["Magnitude"] = feature_df["Impact"].abs()
                        
                        st.dataframe(
                            feature_df,
                            hide_index=True,
                            column_config={
                                "Feature": st.column_config.TextColumn("Word"),
                                "Direction": st.column_config.TextColumn("Direction"),
                                "Magnitude": st.column_config.ProgressColumn(
                                    "Strength",
                                    min_value=0,
                                    max_value=max(feature_df['Magnitude']),
                                    format="%.2f"
                                )
                            }
                        )
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Required models could not be loaded. Please check your installation.")
    elif analyze_button:
        st.warning("⚠️ Please enter a review to analyze.")

# Batch Analysis Tab
with tab2:
    st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
    st.markdown("### 📊 Batch Sentiment Analysis")
    st.markdown("Upload a CSV file with product reviews to analyze multiple reviews at once.")
    
    # Example CSV download
    example_data = pd.DataFrame({
        'review': [
            "This product is amazing! I love everything about it.",
            "Very disappointed with the quality. Not worth the money.",
            "It's okay, but I expected more for the price I paid."
        ]
    })
    
    csv = example_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Example CSV",
        data=csv,
        file_name='example_reviews.csv',
        mime='text/csv',
    )
    
    uploaded_file = st.file_uploader("Upload your CSV file (must contain a 'review' column)", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing your reviews..."):
                df = pd.read_csv(uploaded_file)
                
                if 'review' not in df.columns:
                    st.error("❌ The CSV must contain a column named 'review'.")
                else:
                    # Preprocess and predict
                    with st.status("Processing reviews...") as status:
                        st.write("Cleaning text...")
                        df['cleaned_review'] = df['review'].astype(str).apply(preprocess_text)
                        
                        st.write("Generating predictions...")
                        review_vectors = vectorizer.transform(df['cleaned_review'])
                        proba_scores = model.decision_function(review_vectors)
                        df['sentiment_score'] = proba_scores
                        df['predicted_sentiment'] = ['Positive' if score > 0 else 'Negative' for score in proba_scores]
                        
                        # Calculate confidence
                        df['confidence'] = df['sentiment_score'].apply(
                            lambda x: min(1.0, max(0.0, (x + 1) / 2)) if x > 0 else min(1.0, max(0.0, 1 - (x + 1) / 2))
                        )
                        
                        df['confidence_pct'] = (df['confidence'] * 100).round(1).astype(str) + '%'
                        status.update(label="✅ Processing complete!", state="complete")
                    
                    # Results section
                    st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
                    st.markdown("### 📊 Batch Analysis Results")
                    
                    # Summary statistics
                    total = len(df)
                    positive = df[df['predicted_sentiment'] == 'Positive'].shape[0]
                    negative = df[df['predicted_sentiment'] == 'Negative'].shape[0]
                    pos_percent = (positive / total * 100) if total > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Reviews", total)
                    with col2:
                        st.metric("Positive Reviews", f"{positive} ({pos_percent:.1f}%)")
                    with col3:
                        st.metric("Negative Reviews", f"{negative} ({100-pos_percent:.1f}%)")
                    
                    # Display results
                    st.markdown("#### Review Results")
                    display_df = df[['review', 'predicted_sentiment', 'confidence_pct']]
                    display_df.columns = ['Review', 'Sentiment', 'Confidence']
                    
                    st.dataframe(
                        display_df,
                        column_config={
                            "Review": st.column_config.TextColumn("Review Text"),
                            "Sentiment": st.column_config.TextColumn("Sentiment"),
                            "Confidence": st.column_config.TextColumn("Confidence")
                        }
                    )
                    
                    # Downloadable results
                    csv_download = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Complete Results as CSV",
                        data=csv_download,
                        file_name='sentiment_predictions_detailed.csv',
                        mime='text/csv',
                    )
                    
                    # Visualization
                    st.markdown("#### Sentiment Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sentiment_counts = df['predicted_sentiment'].value_counts()
                    colors = ['#10B981', '#EF4444'] if 'Positive' in sentiment_counts.index else ['#EF4444', '#10B981']
                    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors, ax=ax)
                    plt.title('Sentiment Distribution')
                    plt.ylabel('')
                    
                    # Save to buffer and display
                    buf = BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    buf.seek(0)
                    st.image(buf)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"⚠️ Error processing the file: {str(e)}")

# About Tab
with tab3:
    st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
    st.markdown("### 🔍 About SentiReview Pro")
    st.markdown("""
    **SentiReview Pro** is an advanced sentiment analysis tool designed to help businesses and individuals understand the emotional tone behind product reviews.
    
    #### How It Works
    
    This application uses **machine learning** to determine whether a product review expresses positive or negative sentiment. We employ a Support Vector Machine (SVM) model trained on thousands of product reviews, with TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text into machine-readable features.
    
    #### Key Features
    
    * **Single Review Analysis**: Get instant sentiment predictions for individual reviews
    * **Batch Processing**: Analyze multiple reviews at once by uploading a CSV file
    * **Explainable AI**: Understand what words and phrases influence the sentiment prediction
    * **LIME Visualization**: See highlighted words that contribute to the sentiment
    * **SHAP Analysis**: Deeper understanding of how each word affects the prediction
    
    #### Best Practices
    
    For optimal results:
    * Provide complete, genuine product reviews
    * For batch processing, ensure your CSV has a column named 'review'
    * The more detailed the review, the more accurate and insightful the analysis
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<p class='footer'>•Made By -: Ameen Khan</p>", unsafe_allow_html=True)
