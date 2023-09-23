import pickle
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import face_recognition as fr

# Load known faces from the pickle file
known_face_names, known_face_encodings = pickle.load(open('faces.p', 'rb'))

# Load the input image using OpenCV
image = cv.imread('blackpink.jpg')

# Convert the image from BGR to RGB
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Detect faces using OpenCV
face_locations = fr.face_locations(image_rgb)
face_encodings = fr.face_encodings(image_rgb, face_locations)

# Create an image object for drawing
image_pil = Image.fromarray(image_rgb)
draw = ImageDraw.Draw(image_pil)
font = ImageFont.truetype("arial.ttf", 10)

recogined_info = []

# Iterate through detected faces
for face_encoding, face_location in zip(face_encodings, face_locations):
    top, right, bottom, left = face_location

    # Compare the detected face with known faces
    face_distances = fr.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    match_percentage = (1 - face_distances[best_match_index]) * 100
    name = known_face_names[best_match_index]

    # Draw a rectangle around the detected face
    cv.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)

    name = known_face_names[best_match_index]

    text = f"{name} ({match_percentage:.2f}%)"
    cv.putText(image, text, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

# Display the modified image with OpenCV
cv.imshow('Detected Faces', image)
cv.waitKey(0)
cv.destroyAllWindows()