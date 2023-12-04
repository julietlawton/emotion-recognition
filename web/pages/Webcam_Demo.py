import os
import cv2
import random
import streamlit as st
from emotion_detector import detect_emotion

st.title("Facial Emotion Detector")
st.subheader("AAI 521 Final Project")

picture = st.camera_input("Take a photo")
emotion_label_placeholder = st.empty()

if picture:
    image_file_name = f"webcapture{random.randint(100, 500)}.jpg"
    temp_file_path = os.path.join("temp", image_file_name)
    with open(temp_file_path, "wb") as f: 
        f.write(picture.getbuffer())

    img = cv2.imread(temp_file_path)
    emotion, confidence = detect_emotion(img)
    markdown = f"**Predicted Emotion:** {emotion} ({confidence})"
    emotion_label_placeholder.markdown(markdown)
