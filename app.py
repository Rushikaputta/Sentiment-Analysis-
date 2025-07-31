import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline

# Load emotion detection model
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

st.set_page_config(page_title="Emotion Classifier", layout="centered")
st.title("🎯 Emotion Classifier App")
st.write("Detect emotion from any text using a pre-trained NLP model.")

# Input box
user_input = st.text_area("✍️ Type your text here:", 
    "I am happy that I’ve completed projects in Machine Learning and achieved good accuracy!")

if st.button("🔍 Submit"):
    with st.spinner("Analyzing..."):
        result = classifier(user_input)[0]
        df = pd.DataFrame(result)
        df['label'] = df['label'].str.lower()
        df = df.sort_values(by="score", ascending=False)

        # Show top emotion
        top_emotion = df.iloc[0]
        emoji_dict = {
            "joy": "😊", "sadness": "😢", "anger": "😠",
            "fear": "😨", "disgust": "🤢", "neutral": "😐",
            "surprise": "😲", "shame": "😳"
        }
        st.success(f"**Emotion Detected:** `{top_emotion['label'].capitalize()}` {emoji_dict.get(top_emotion['label'], '')}")
        st.markdown(f"**Confidence:** `{top_emotion['score']:.2f}`")

        # Plot probabilities
        st.subheader("📊 Emotion Probabilities")
        st.bar_chart(df.set_index("label")["score"])
