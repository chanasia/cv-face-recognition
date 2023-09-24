import pickle
import cv2 as cv
import numpy as np
import face_recognition as fr
from tkinter.filedialog import askopenfilename

# Load known faces from the pickle file
known_face_names, known_face_encodings = pickle.load(open('models/faces.p', 'rb'))

# Load the input image using OpenCV
load_image = askopenfilename()
image = cv.imread(load_image)

# Convert the image from BGR to RGB
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Detect faces using OpenCV
face_locations = fr.face_locations(image_rgb, model="hog")
face_encodings = fr.face_encodings(image_rgb, face_locations)

recognized_names = []
recognized_infos = []
recognized_percent = []

# Iterate through detected faces
for face_encoding, face_location in zip(face_encodings, face_locations):
    top, right, bottom, left = face_location

    # Compare the detected face with known faces
    face_distances = fr.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    match_percentage = (1 - face_distances[best_match_index]) * 100
    name = known_face_names[best_match_index]

    cv.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)

    if match_percentage > 60:
        if name not in recognized_names:
            recognized_names.append(name)
            recognized_infos.append((top, right, bottom, left))
            recognized_percent.append(match_percentage)
        else:
            idx_same_name = recognized_names.index(name)
            if match_percentage > recognized_percent[idx_same_name]:
                #แก้ไขอันชื่อเก่าเป็น unknown
                recognized_names[idx_same_name] = "Unknown"
                recognized_percent[idx_same_name] = 0
                
                #เพิ่มใบหน้าใหม่
                recognized_names.append(name)
                recognized_infos.append((top, right, bottom, left))
                recognized_percent.append(match_percentage)
            else:
                recognized_names.append("Unknown")
                recognized_infos.append((top, right, bottom, left))
                recognized_percent.append(0)
    else:
        recognized_names.append("Unknown")
        recognized_infos.append((top, right, bottom, left))
        recognized_percent.append(0)

for name, positions, percent in zip(recognized_names, recognized_infos, recognized_percent):
    # Draw a rectangle around the detected face
    top, right, bottom, left = positions
    percent_text = "" if percent == 0 else f" ({percent:.1f}%)"
    text = f"{name}{percent_text}"
    cv.putText(image, text, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

# Display the modified image with OpenCV
cv.imshow('Detected Faces', image)
cv.waitKey(0)
cv.destroyAllWindows()