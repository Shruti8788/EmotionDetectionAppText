import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
import pickle
import nltk
import matplotlib.pyplot as plt

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# Load models
lg = pickle.load(open('logistic_regresion.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))
lb = pickle.load(open('label_encoder.pkl','rb'))

# Clean text
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# Predict emotion
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    probability = np.max(lg.predict_proba(input_vectorized))
    probs = lg.predict_proba(input_vectorized)[0]

    return predicted_emotion, probability, probs


# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Emotion AI", page_icon="🤖", layout="wide")

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #eef2f3, #8e9eab);
}
.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #2c3e50;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.sidebar.title("🤖 Emotion AI")
st.sidebar.write("Detect human emotions from text using Machine Learning.")
st.sidebar.write("Supported Emotions:")
st.sidebar.write("😊 Joy")
st.sidebar.write("😨 Fear")
st.sidebar.write("😡 Anger")
st.sidebar.write("❤️ Love")
st.sidebar.write("😢 Sadness")
st.sidebar.write("😲 Surprise")

# ================== TITLE ==================
st.markdown('<p class="title">Emotion Detection AI</p>', unsafe_allow_html=True)

# ================== LAYOUT ==================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Enter Text")
    user_input = st.text_area("", height=200)

    if st.button("Predict Emotion"):
        emotion, prob, probs = predict_emotion(user_input)

        st.markdown("### Prediction Result")
        st.success(f"Emotion: {emotion}")
        st.progress(int(prob * 100))
        st.write(f"Confidence: {round(prob*100,2)} %")

        emoji_dict = {
            "joy": "😄",
            "fear": "😨",
            "anger": "😡",
            "love": "❤️",
            "sadness": "😢",
            "surprise": "😲"
        }

        if emotion in emoji_dict:
            st.markdown(f"# {emoji_dict[emotion]}")

with col2:
    st.markdown("### Emotion Probability Chart")

    if 'probs' in locals():
        emotions = lb.classes_

        fig = plt.figure()
        plt.bar(emotions, probs)
        plt.xlabel("Emotions")
        plt.ylabel("Probability")
        plt.title("Emotion Prediction Probability")
        st.pyplot(fig)