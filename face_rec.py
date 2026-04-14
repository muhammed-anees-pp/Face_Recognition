import glob
import os
import cv2
import face_recognition
import numpy as np

class Facerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            encodings = face_recognition.face_encodings(rgb_img)

            if len(encodings) == 0:
                print(f"No face found in {img_path}, skipping...")
                continue

            img_encoding = encodings[0]

            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

    def add_new_face(self, name, img_path):
        """
        Dynamically loads an image, encodes it, and adds it to the face records.
        Returns True if successful, False if no face was found in the image.
        """
        img = cv2.imread(img_path)
        if img is None:
            return False
            
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)

        if len(encodings) == 0:
            print(f"[WARNING] No face found in {img_path}, unable to register.")
            # Optionally remove the invalid image
            if os.path.exists(img_path):
                os.remove(img_path)
            return False

        self.known_face_encodings.append(encodings[0])
        self.known_face_names.append(name)
        return True
