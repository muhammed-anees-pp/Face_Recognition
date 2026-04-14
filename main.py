import cv2
from face_rec import Facerec

fr = Facerec()
fr.load_encoding_images("images/")

cap = cv2.VideoCapture(0)

if not cap.isOpened(): 
    print("Error: Unable to open the camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Unable to read frame.")
        break

    face_locations, face_names = fr.detect_known_faces(frame)
    for face_locations, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_locations[0], face_locations[1], face_locations[2], face_locations[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.putText(frame, "Press 'r' to register a new face", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('r'):
        name = ""
        frame_snapshot = frame.copy()
        while True:
            display_frame = frame_snapshot.copy()
            
            # Create a dark overlay
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            cv2.putText(display_frame, "--- Registration Mode ---", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Enter Name: {name}_", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press ENTER to save, ESC to cancel", (50, 150), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("Frame", display_frame)
            k = cv2.waitKey(0)
            
            if k == 13: # Enter
                if name.strip():
                    img_path = f"images/{name.strip()}.jpg"
                    # Pass the clean frame_snapshot (without bounding boxes if there was an unknown face)
                    cv2.imwrite(img_path, frame_snapshot)
                    success = fr.add_new_face(name.strip(), img_path)
                    if success:
                        print(f"[INFO] Registered {name.strip()} successfully!")
                    else:
                        print(f"[WARNING] No face found. Registration failed.")
                break
            elif k == 27: # Esc
                break
            elif k in (8, 127): # Backspace
                name = name[:-1]
            elif 32 <= k <= 126:
                name += chr(k)

cap.release()
cv2.destroyAllWindows()