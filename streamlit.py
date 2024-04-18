# prompt: an app which converts normal video with sign language to subtitles

# Import necessary libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("sign_language_model.h5")

# Define a dictionary to map integer predictions to characters
word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Define a function to preprocess the image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    img = np.array(img) / 255.0
    return img

# Define a function to predict the sign language
def predict_sign_language(img):
    img = preprocess_image(img)
    prediction = model.predict(np.array([img]))
    predicted_class = np.argmax(prediction)
    return word_dict[predicted_class]

# Define a function to capture video and generate subtitles
def generate_subtitles():
    cap = cv2.VideoCapture(0)
    subtitles = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        predicted_character = predict_sign_language(frame)
        subtitles += predicted_character
        cv2.imshow('Sign Language Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return subtitles

# Generate subtitles
subtitles = generate_subtitles()

# Print the generated subtitles
print(subtitles)
