import streamlit as st
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
from keras.models import load_model
from extract_features import extract_features
import librosa
from PIL import Image

# Load the trained model
model = load_model("emotion_model.h5")
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def predict_emotion(audio_path):
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    return emotions[np.argmax(prediction)]

# 🎯 Streamlit UI
st.set_page_config(layout="wide")

st.sidebar.title("Emotion Recognition")

image = Image.open("img2.jpg")  #left-sidebar image
st.sidebar.image(image, use_container_width=True)


st.sidebar.markdown("""
    <h2 style='font-family:Arial; color:#3366cc;'>Zeeshan Ali</h2>
    <p style='margin-bottom:5px;'>University of Malakand</p>
    <p style='margin-bottom:10px;'>Machine Learning Project</p>

    <p style='margin: 5px 0;'>
        <a href='https://www.linkedin.com/in/alizeeshanse' target='_blank' style='text-decoration: none;'>
            <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='20' style='vertical-align:middle; margin-right:5px;'>
            LinkedIn
        </a>
    </p>

    <p style='margin: 5px 0;'>
        <a href='https://github.com/alizeeshan-se' target='_blank' style='text-decoration: none;'>
            <img src='https://cdn-icons-png.flaticon.com/512/25/25231.png' width='20' style='vertical-align:middle; margin-right:5px;'>
            GitHub
        </a>
    </p>

    <p style='margin: 5px 0;'>
        <img src='https://cdn-icons-png.flaticon.com/512/281/281769.png' width='20' style='vertical-align:middle; margin-right:5px;'>
        <a href='mailto:alizeeshanse@gmail.com' style='text-decoration: none; color: white;'>alizeeshanse@gmail.com</a>
    </p>

    <p style='margin: 5px 0;'>
        <img src='https://cdn-icons-png.flaticon.com/512/733/733585.png' width='20' style='vertical-align:middle; margin-right:5px;'>
        <a href='https://wa.me/923499373126' target='_blank' style='text-decoration: none; color: white;'>+92 349 9373126</a>
    </p>

    <hr style='margin:10px 0; border: 0; border-top: 1px solid #ccc;'>

    <p style='color:white; font-weight:bold;'>✅ Available for Work</p>
""", unsafe_allow_html=True)



st.markdown("""
    <h1 style='font-family: Arial; color: #1a75ff;'>🎙️ Emotion Recognition From Speech</h1>
    <p style='font-size:16px;'>Upload a <code>.wav</code> file <strong>or</strong> record your voice to predict the emotion.</p>
""", unsafe_allow_html=True)


# --- File Upload Option ---
st.markdown("<p style='font-size:16px; color:#1a75ff; font-family:Verdana;'>📤 <strong>Upload your voice (.wav)</strong></p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["wav"])


if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    try:
        prediction = predict_emotion(uploaded_file)
        st.success(f"🎯 Predicted Emotion: **{prediction.upper()}**")
    except Exception as e:
        st.error("Error processing the uploaded audio.")
        st.error(str(e))

        
