import pickle
import face_recognition as fr
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import concurrent.futures

# Load known face data from pickle file
known_face_names, known_face_encodings = pickle.load(open('faces.p', 'rb'))

# Initialize the font for drawing text
font = ImageFont.truetype("arial.ttf", 12)

def process_frame(frame):
    # Convert the OpenCV frame to an RGB image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find face locations in the frame
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    # Create a PIL Image from the frame
    pil_image = Image.fromarray(frame)

    # Create a draw object to annotate the image
    draw = ImageDraw.Draw(pil_image)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        face_distances = fr.face_distance(known_face_encodings, face_encoding)

        top, right, bottom, left = face_location
        draw.rectangle([left, top, right, bottom], outline=(0, 255, 0))

        best_match_index = np.argmin(face_distances)

        match_percentage = (1 - face_distances[best_match_index]) * 100

        name = known_face_names[best_match_index]
        text = f"{name} ({match_percentage:.2f}%)"
        # text = ""
        # if match_percentage < 63:
        #     text = "unknown"
        # else:
        #     name = known_face_names[best_match_index]
        #     text = f"{name} ({match_percentage:.2f}%)"

        draw.text((left, top), text, fill=(0, 0, 255), font=font)

    return np.array(pil_image)

# Initialize the webcam (you can change the camera index as needed)
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        result_frame = executor.submit(process_frame, frame)
        result_frame = result_frame.result()

    cv2.imshow('Video', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()