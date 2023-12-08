import numpy as np
import tensorflow as tf
import cv2
import imutils
import face_detection

fer_model = tf.keras.models.load_model("web/models/modelv3.keras")

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

    cropped_image = image[max(y, 0):min(y + h, image.shape[0]), max(x, 0):min(x + w, image.shape[1])]
    resized_image = cv2.resize(cropped_image, (96, 96))

    image = resized_image.reshape(96, 96, 1).astype('float32')/255.0

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
# test_image = cv2.imread('web/temp/webcapture125.jpg')
# test_image = cv2.imread('web/temp/sad_man.jpeg')
# test_bbox = face_detection.get_bounding_box(test_image)
# img = preprocess_image(test_image, test_bbox)
# emotion_label, confidence_level = detect_emotion(test_image, test_bbox)
# print(emotion_label, confidence_level)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()