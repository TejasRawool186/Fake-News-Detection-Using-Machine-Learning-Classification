import streamlit as st
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# -----------------------------
# Rainbow Title Styling
# -----------------------------
st.markdown(
"""
<h1 style="
text-align:center;
font-size:48px;
background: linear-gradient(90deg, red, orange, yellow, green, blue, indigo, violet);
-webkit-background-clip: text;
color: transparent;
font-weight: bold;">
📰 Fake News Detection System
</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"<p style='text-align:center; font-size:18px;'>Detect whether a news article is <b>Fake</b> or <b>Real</b> using multiple Machine Learning models.</p>",
unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# Load Models
# -----------------------------
lr_model = joblib.load("logistic_regression_model.pkl")
nb_model = joblib.load("naive_bayes_model.pkl")
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# Sidebar (Model Performance)
# -----------------------------
st.sidebar.title("📊 Model Performance")

st.sidebar.success("SVM Accuracy: 99.5%")
st.sidebar.info("Logistic Regression: 98.3%")
st.sidebar.warning("Naive Bayes: 93.7%")

st.sidebar.markdown("---")

# -----------------------------
# User Input
# -----------------------------
news_input = st.text_area(
    "Paste a News Article or Headline:",
    height=200,
    placeholder="Example: Government announces new healthcare policy..."
)

# -----------------------------
# Prediction Functions
# -----------------------------
def show_result(model, pred):
    if pred == 1:
        st.success(f"{model}\n\n🟢 Real News")
    else:
        st.error(f"{model}\n\n🔴 Fake News")

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("🔍 Analyze News"):

    if news_input.strip() == "":
        st.warning("⚠️ Please enter news text to analyze.")

    else:

        with st.spinner("Analyzing News..."):

            text_vector = vectorizer.transform([news_input])

            lr_pred = lr_model.predict(text_vector)[0]
            nb_pred = nb_model.predict(text_vector)[0]
            svm_pred = svm_model.predict(text_vector)[0]

        st.markdown("## 🤖 Model Predictions")

        col1, col2, col3 = st.columns(3)

        with col1:
            show_result("Logistic Regression", lr_pred)

        with col2:
            show_result("Naive Bayes", nb_pred)

        with col3:
            show_result("SVM", svm_pred)

        # -----------------------------
        # Majority Voting
        # -----------------------------
        votes = [lr_pred, nb_pred, svm_pred]
        final = max(set(votes), key=votes.count)

        st.markdown("---")
        st.markdown("## 🧠 Final Decision")

        if final == 1:
            st.success("🟢 Real News (Majority Voting)")
        else:
            st.error("🔴 Fake News (Majority Voting)")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Fake News Detection System using NLP and Machine Learning")