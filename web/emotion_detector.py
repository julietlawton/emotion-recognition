import numpy as np
import tensorflow as tf
import cv2
import imutils

fer_model = tf.keras.models.load_model("models/test_model.keras")

emotion_mapping = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

def preprocess_image(image, bbox):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    x, y, w, h = bbox
    face_centerpoint = (x+w/2, y+h/2)
    #image = imutils.resize(image_gray, width=192)
    #image = cv2.resize(image_gray, (250, 192)).astype('float32')/255.0
    #print(bbox, face_centerpoint)

    height, width = image.shape[:2]
    remove_vertical = height - 192
    remove_horizontal = width - 192
    top_trim = remove_vertical // 2
    bottom_trim = remove_vertical - top_trim
    left_trim = remove_horizontal // 2
    right_trim = remove_horizontal - left_trim

    image = image[top_trim:height - bottom_trim, left_trim:width - right_trim]

    image = image.reshape(192, 192, 1).astype('float32')/255.0
    image = np.repeat(image, 3, axis=2)

    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    image_tensor = np.expand_dims(image_tensor, axis=0)
    return image_tensor

def detect_emotion(image, bbox):
    preprocessed_input = preprocess_image(image, bbox)
    prediction = fer_model.predict(preprocessed_input)

    predicted_emotion = emotion_mapping[np.argmax(prediction)]
    confidence_level = np.max(prediction)
    rounded = np.round(confidence_level, 2)
    confidence_str = "{:.2f}".format(rounded)

    return predicted_emotion, confidence_str

# Debug
# test_image = cv2.imread('temp/sad_man.jpeg')
# test_bbox = (249, 113, 136, 136)
# img = preprocess_image(test_image, test_bbox)
# emotion_label, confidence_level = detect_emotion(test_image)
# print(emotion_label, confidence_level)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()