import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier('web/models/haarcascade_frontalface_alt.xml')

def get_bounding_box(image):
   image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   bbox = face_cascade.detectMultiScale(image_gray, 1.1, 2)[0]
   return bbox

def draw_bounding_box(image, bbox, emotion_label=None, confidence_level=None):
   x, y, w, h = bbox
   cv2.rectangle(image, (x,y), (x+w,y+h), (0, 0, 255), 2)

   if emotion_label is not None:
      label_pos = (x, y+-5)
      label = emotion_label if confidence_level is None else f"{emotion_label} {confidence_level}"
      cv2.putText(
        image, 
        label, 
        label_pos, 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=0.7, 
        color=(0, 0, 255), 
        thickness=2
    )
   return image

# Debug
# img = cv2.imread("smiling_man_placeholder.jpg")
# img, bbox = get_bounding_box(img, "Happy", 0.88)
# cv2.imshow(img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()