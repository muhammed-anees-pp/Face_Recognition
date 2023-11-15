import cv2
from face_rec import Facerec

# Taking images
fr = Facerec()
fr.load_encoding_images("images/")

# Camera
cap = cv2.VideoCapture(0)

# If not opened the camera
if not cap.isOpened(): 
    print("Error: Unable to open the camera.")
    exit()

# Loop for face recognition 
while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Unable to read frame.")
        break

    # Face detection
    face_locations, face_names = fr.detect_known_faces(frame)
    for face_locations, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_locations[0], face_locations[1], face_locations[2], face_locations[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27: # Esc key used to trigger breaking the program, 27 corresponds to the ASCII value of the 'Esc' key.
        break

cap.release()
cv2.destroyAllWindows()