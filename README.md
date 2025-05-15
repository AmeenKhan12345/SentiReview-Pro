# Sentiment Analysis of Product Reviews with Explainable AI (XAI)

**A Streamlit web app that classifies Amazon product reviews into Positive and Negative sentiments using TF-IDF + SVM, with explainability via LIME and SHAP.**

---

## 🚀 Live Demo

[Streamlit App URL](https://share.streamlit.io/your-user/your-repo/app.py)

---

## 📋 Table of Contents

1. [Features](#features)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Model Training](#model-training)
7. [Explainability](#explainability)
8. [Future Work](#future-work)
9. [License](#license)

---

## 🌟 Features

* **Single Review Prediction**: Input a review and get Positive/Negative sentiment.
* **Batch Processing**: Upload a CSV of reviews, process in bulk, and download annotated results.
* **Explainable AI**:

  * **LIME**: Highlights top words influencing each prediction.
  * **SHAP**: Visualizes feature impact on model decisions.
* **Confidence Score**: Displays prediction confidence with a progress bar.
* **Interactive UI**: Built with Streamlit for a user-friendly experience.

---

## 📦 Dataset

* **Source**: Amazon Product Review Dataset
* **Training Size**: 1,000,000+ reviews
* **Test Size**: 400,000+ reviews
* **Classes**: Positive, Negative

---

## 🗂️ Project Structure

```plaintext
repo-root/
├── app.py                    # Streamlit application script
├── svm_sentiment_model.pkl   # Trained SVM model
├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/AmeenKhan12345/SentiReview-Pro.git
   cd SentiReview-Pro
   ```

2. **Create a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**:

   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## 🛠️ Usage

**Run the Streamlit app**:

```bash
streamlit run app.py
```

* Open `http://localhost:8501` in your browser.
* Enter a review in the text area to get sentiment and explanations.
* Expand the LIME/SHAP panels for insights.
* Upload a CSV file under "Batch Sentiment Prediction" to process multiple reviews.

---

## 🔄 Model Training

*(Optional: If you want to retrain)*

1. **Prepare data**: Ensure `product_reviews_train.csv` and `product_reviews_test.csv` are in the root.
2. **Modify and run** `train_model.py` (or similar notebook) to preprocess, vectorize, and train the SVM.
3. **Save artifacts**:

   ```python
   joblib.dump(model, 'svm_sentiment_model.pkl')
   joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
   ```

---

## 🔍 Explainability

* **LIME**:

  * Highlights top 10 words contributing to sentiment.
  * Integrated via `explain_prediction` in `app.py`.

* **SHAP**:

  * Uses `shap.LinearExplainer` on a representative background dataset.
  * Summary plots displayed under the SHAP expander.

---

## 🔮 Future Work

* Multi-class sentiment (include Neutral)
* Fine-tune transformer models (BERT, RoBERTa)
* Real-time monitoring dashboard
* Multilingual support (code-mixed reviews)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
