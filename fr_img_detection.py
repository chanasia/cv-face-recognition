import pickle
import face_recognition as fr
import numpy as np
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageDraw, ImageFont

known_face_names, known_face_encodings = pickle.load(open('faces.p', 'rb'))

image = Image.open('blackpink.jpg')
face_locations = fr.face_locations(np.array(image))
face_encodings = fr.face_encodings(np.array(image), face_locations)

draw = ImageDraw.Draw(image)

font = ImageFont.truetype("arial.ttf", 10)  

faces = zip(face_encodings, face_locations)

tmp_names = []
tmp_positions = []

for idx, face in enumerate(faces):
    face_encoding, face_location = face
    face_distances = fr.face_distance(known_face_encodings, face_encoding)
    top, right, bottom, left = face_location 

    best_match_index = np.argmin(face_distances)

    match_percentage = (1 - face_distances[best_match_index]) * 100
    name = known_face_names[best_match_index]

    draw.rectangle([left, top, right, bottom], outline=(0, 255, 0))
    name = known_face_names[best_match_index]
    text = f"{name} ({match_percentage:.2f}%)"
    draw.text((left, top), text, fill=(0, 0, 255), font=font)

image.show()