import streamlit as st
import joblib

# Load model
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

st.title("📰 Fake News Detection App")

text = st.text_area("Enter News Text")

if st.button("Predict"):
    if text.strip():
        vec = vectorizer.transform([text])
        result = model.predict(vec)

        if result[0] == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")
    else:
        st.warning("Enter some text")