import face_recognition
import cv2
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load images and encode known faces
known_faces = []
known_names = []

print(current_dir)

# Load and encode known faces
known_faces.append(face_recognition.load_image_file(os.path.join(current_dir, "images/student1.jpg")))
known_faces.append(face_recognition.load_image_file(os.path.join(current_dir, "images/student2.jpg")))
known_faces.append(face_recognition.load_image_file(os.path.join(current_dir, "images/student3.jpg")))


known_names = ["Nevin","sreejith","binn"]

# Encode known faces
known_encodings = []
for img in known_faces:
    face_encoding = face_recognition.face_encodings(img)
    if len(face_encoding) > 0:
        known_encodings.append(face_encoding[0])
    else:
        print("No face found in one of the images.")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face matches any of the known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding)

        name = "Unknown"

        # If a match is found, use the name of the matched known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # If 'q' is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
