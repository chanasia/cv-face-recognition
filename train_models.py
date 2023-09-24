import os
import pickle
import face_recognition as fr
import numpy as np

if not os.path.exists("models"):
    os.makedirs("models")

list_known_faces = []

for person in os.listdir("known_faces"):
  for image_name in os.listdir(os.path.join("known_faces", person)):
    list_known_faces.append((person, image_name))

known_face_names = []
known_face_encodings = []

for face_info in list_known_faces:
    path_filename = os.path.join("known_faces", face_info[0], face_info[1])
    face_image = fr.load_image_file(path_filename)
    face_encodings = fr.face_encodings(face_image, model="cnn")

    # Ensure that at least one face encoding was computed for the image
    if face_encodings:
        known_face_encodings.append(face_encodings[0])
        known_face_names.append(face_info[0])
    else:
        print(f"No face detected in {path_filename}")

pickle.dump((known_face_names, known_face_encodings), open('models/faces.p', 'wb')) # save model