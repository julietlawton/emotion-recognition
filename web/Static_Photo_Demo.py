import os
import cv2
import streamlit as st
from PIL import Image
from face_detection import get_bounding_box, draw_bounding_box
from emotion_detector import detect_emotion

st.title("Facial Emotion Detector")
st.subheader("AAI 521 Final Project")


image_file = st.file_uploader(
    "Choose an image", 
    type=['png', 'jpg'], 
    accept_multiple_files=False, 
    help="Upload the image you want to test. Image must be bigger than 192x192 pixels.", 
    on_change=None,
    label_visibility="visible"
)

if image_file is not None:
    temp_file_path = os.path.join("temp", image_file.name)
    with open(temp_file_path, "wb") as f: 
        f.write(image_file.getbuffer())         
    img = cv2.imread(temp_file_path)

    if img.shape[0] > 192 and img.shape[1] > 192:
        bbox = get_bounding_box(img)
        emotion, confidence = detect_emotion(img, bbox)
        img = draw_bounding_box(img, bbox, emotion_label=emotion, confidence_level=confidence)

        st.image(img, width=700, channels="BGR")
        st.success("Success")
    else:
        st.error("Image is too small. Please upload an image bigger than 192x192 pixels.")
else:
    img = cv2.imread("web/smiling_man_placeholder.jpg")
    # bbox = get_bounding_box(img)
    # img = draw_bounding_box(img, bbox, emotion_label="Happy", confidence_level=0.88)
    st.image(img, width=700, channels="BGR")