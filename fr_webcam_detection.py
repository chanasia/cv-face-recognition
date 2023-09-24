import pickle
import cv2 as cv
import numpy as np
import face_recognition as fr
import concurrent.futures

# Load known faces from the pickle file
known_face_names, known_face_encodings = pickle.load(open('models/faces.p', 'rb'))

def process_detect_face(face_locations, face_encodings):
    recognized_names = []
    recognized_infos = []
    recognized_percent = []
    
    for face_encoding, face_location in zip(face_encodings, face_locations):
        top, right, bottom, left = face_location

        # Compare the detected face with known faces
        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        match_percentage = (1 - face_distances[best_match_index]) * 100
        name = known_face_names[best_match_index]

        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

        if match_percentage > 50:
            if name not in recognized_names:
                recognized_names.append(name)
                recognized_infos.append((top, right, bottom, left))
                recognized_percent.append(match_percentage)
            else:
                idx_same_name = recognized_names.index(name)
                if match_percentage > recognized_percent[idx_same_name]:
                    #แก้ไขอันชื่อเก่าเป็น unknown
                    recognized_names[idx_same_name] = "unknown"
                    recognized_percent[idx_same_name] = 0
                    
                    #เพิ่มใบหน้าใหม่
                    recognized_names.append(name)
                    recognized_infos.append((top, right, bottom, left))
                    recognized_percent.append(match_percentage)
                else:
                    recognized_names = "unknown"
                    recognized_infos.append((top, right, bottom, left))
                    recognized_percent.append(0)
        else:
            recognized_names = "unknown"
            recognized_infos.append((top, right, bottom, left))
            recognized_percent.append(0)
    
    # Draw a rectangle around the detected face
    for name, positions, percent in zip(recognized_names, recognized_infos, recognized_percent):
        # Draw a rectangle around the detected face
        top, right, bottom, left = positions
        percent_text = "" if percent == 0 else f" ({percent:.2f}%)"
        text = f"{name}{percent_text}"
        cv.putText(frame, text, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

# Open a connection to the camera (0 is usually the default camera)
video_capture = cv.VideoCapture(0)

# Check if the camera is opened successfully
if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame from BGR to RGB
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Detect faces using face_recognition
    face_locations = fr.face_locations(frame_rgb, model="hog")
    face_encodings = fr.face_encodings(frame_rgb, face_locations)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        result_frame = executor.submit(process_detect_face, face_locations, face_encodings)
        result_frame = result_frame.result()
    
    # Display the frame with OpenCV
    cv.imshow('Real-time Face Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
video_capture.release()
cv.destroyAllWindows()